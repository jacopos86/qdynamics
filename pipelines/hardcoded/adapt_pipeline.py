#!/usr/bin/env python3
"""Hardcoded ADAPT-VQE end-to-end Hubbard / Hubbard-Holstein pipeline.

Flow:
1) Build Hubbard (or HH) Hamiltonian (JW) from repo source-of-truth helpers.
2) Build operator pool (UCCSD, CSE, full_hamiltonian, HVA, or PAOP variants).
3) Run standard ADAPT-VQE: commutator gradients, one operator per
   iteration, COBYLA/SPSA inner optimizer, optional repeats.
4) Run Suzuki-2 Trotter dynamics + exact dynamics from the ADAPT ground state.
5) Emit JSON + compact PDF artifact.

Uses the *same* src/quantum/ primitives as the regular VQE hardcoded pipeline.
No dependency on Qiskit in the core path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — this file lives at pipelines/hardcoded/adapt_pipeline.py
# REPO_ROOT is the top-level Holstein_test directory.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]  # Holstein_test/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docs.reports.pdf_utils import (
    HAS_MATPLOTLIB,
    require_matplotlib,
    get_plt,
    get_PdfPages,
    render_text_page,
    current_command_string,
)

# Module-level aliases used by the plotting body
plt = get_plt() if HAS_MATPLOTLIB else None  # type: ignore[assignment]
PdfPages = get_PdfPages() if HAS_MATPLOTLIB else type("PdfPages", (), {})  # type: ignore[misc]

# ---------------------------------------------------------------------------
# Imports from the active repo quantum modules (no pydephasing fallback).
# ---------------------------------------------------------------------------
from src.quantum.hubbard_latex_python_pairs import (
    build_hubbard_hamiltonian,
    build_hubbard_holstein_hamiltonian,
    boson_qubits_per_site,
)
from src.quantum.hartree_fock_reference_state import hartree_fock_statevector
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.compiled_polynomial import (
    CompiledPolynomialAction,
    adapt_commutator_grad_from_hpsi,
    apply_compiled_polynomial as _apply_compiled_polynomial_shared,
    compile_polynomial_action as _compile_polynomial_action_shared,
    energy_via_one_apply,
)
from src.quantum.pauli_actions import (
    CompiledPauliAction,
    apply_compiled_pauli as _apply_compiled_pauli_shared,
    apply_exp_term as _apply_exp_term_shared,
    compile_pauli_action_exyz as _compile_pauli_action_exyz_shared,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.pauli_words import PauliTerm
from src.quantum.spsa_optimizer import spsa_minimize
from src.quantum.vqe_latex_python_pairs import (
    AnsatzTerm,
    HardcodedUCCSDAnsatz,
    HubbardHolsteinLayerwiseAnsatz,
    HubbardTermwiseAnsatz,
    apply_exp_pauli_polynomial,
    apply_pauli_string,
    basis_state,
    exact_ground_energy_sector,
    exact_ground_energy_sector_hh,
    expval_pauli_polynomial,
    half_filled_num_particles,
    hamiltonian_matrix,
    hartree_fock_bitstring,
    hubbard_holstein_reference_state,
)
from pipelines.hardcoded.hh_continuation_types import (
    CandidateFeatures,
    Phase2OptimizerMemoryAdapter,
    ScaffoldFingerprintLite,
)
from pipelines.hardcoded.hh_continuation_generators import (
    build_pool_generator_registry,
    build_runtime_split_children,
    build_split_event,
    selected_generator_metadata_for_labels,
)
from pipelines.hardcoded.hh_continuation_motifs import (
    extract_motif_library,
    load_motif_library_from_json,
    select_tiled_generators_from_library,
)
from pipelines.hardcoded.hh_continuation_symmetry import (
    build_symmetry_spec,
    leakage_penalty_from_spec,
    verify_symmetry_sequence,
)
from pipelines.hardcoded.hh_continuation_rescue import (
    RescueConfig,
    rank_rescue_candidates,
    should_trigger_rescue,
)
from pipelines.hardcoded.hh_continuation_stage_control import (
    StageController,
    StageControllerConfig,
    allowed_positions,
    detect_trough,
    should_probe_positions,
)
from pipelines.hardcoded.hh_continuation_scoring import (
    CompatibilityPenaltyOracle,
    FullScoreConfig,
    MeasurementCacheAudit,
    Phase2CurvatureOracle,
    Phase1CompileCostOracle,
    Phase2NoveltyOracle,
    SimpleScoreConfig,
    build_candidate_features,
    build_full_candidate_features,
    greedy_batch_select,
    shortlist_records,
)
from pipelines.hardcoded.hh_continuation_pruning import (
    PruneConfig,
    apply_pruning,
    post_prune_refit,
    rank_prune_candidates,
)

try:
    from src.quantum.operator_pools import make_pool as make_paop_pool
except Exception as exc:  # pragma: no cover - defensive fallback
    make_paop_pool = None
    _PAOP_IMPORT_ERROR = str(exc)
else:
    _PAOP_IMPORT_ERROR = ""

EXACT_LABEL = "Exact_Hardcode"
EXACT_METHOD = "python_matrix_eigendecomposition"
_ADAPT_GRADIENT_PARITY_RTOL = 1e-8

PAULI_MATS = {
    "e": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
    "x": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


def _ai_log(event: str, **fields: Any) -> None:
    payload = {
        "event": str(event),
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    print(f"AI_LOG {json.dumps(payload, sort_keys=True, default=str)}", flush=True)


# ---------------------------------------------------------------------------
# Utility helpers (mirror hardcoded VQE pipeline)
# ---------------------------------------------------------------------------

def _to_ixyz(label_exyz: str) -> str:
    return str(label_exyz).replace("e", "I").upper()


def _normalize_state(psi: np.ndarray) -> np.ndarray:
    nrm = float(np.linalg.norm(psi))
    if nrm <= 0.0:
        raise ValueError("Encountered zero-norm state.")
    return psi / nrm


def _collect_hardcoded_terms_exyz(poly: Any, tol: float = 1e-12) -> tuple[list[str], dict[str, complex]]:
    coeff_map: dict[str, complex] = {}
    order: list[str] = []
    for term in poly.return_polynomial():
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if abs(coeff) <= tol:
            continue
        if label not in coeff_map:
            coeff_map[label] = 0.0 + 0.0j
            order.append(label)
        coeff_map[label] += coeff
    cleaned_order = [lbl for lbl in order if abs(coeff_map[lbl]) > tol]
    cleaned_map = {lbl: coeff_map[lbl] for lbl in cleaned_order}
    return cleaned_order, cleaned_map


def _pauli_matrix_exyz(label: str) -> np.ndarray:
    mats = [PAULI_MATS[ch] for ch in label]
    out = mats[0]
    for mat in mats[1:]:
        out = np.kron(out, mat)
    return out


def _build_hamiltonian_matrix(coeff_map_exyz: dict[str, complex]) -> np.ndarray:
    if not coeff_map_exyz:
        return np.zeros((1, 1), dtype=complex)
    nq = len(next(iter(coeff_map_exyz)))
    dim = 1 << nq
    hmat = np.zeros((dim, dim), dtype=complex)
    for label, coeff in coeff_map_exyz.items():
        hmat += coeff * _pauli_matrix_exyz(label)
    return hmat


# ---------------------------------------------------------------------------
# Compiled Pauli + Trotter (identical to VQE pipeline)
# ---------------------------------------------------------------------------

def _compile_pauli_action(label_exyz: str, nq: int) -> CompiledPauliAction:
    return _compile_pauli_action_exyz_shared(label_exyz=label_exyz, nq=nq)


def _apply_compiled_pauli(psi: np.ndarray, action: CompiledPauliAction) -> np.ndarray:
    return _apply_compiled_pauli_shared(psi=psi, action=action)


def _compile_polynomial_action(
    poly: Any,
    tol: float = 1e-15,
    *,
    pauli_action_cache: dict[str, CompiledPauliAction] | None = None,
) -> CompiledPolynomialAction:
    """Compile a PauliPolynomial into reusable Pauli actions for repeated apply."""
    terms = poly.return_polynomial()
    if not terms:
        return CompiledPolynomialAction(nq=0, terms=tuple())
    return _compile_polynomial_action_shared(
        poly,
        tol=float(tol),
        pauli_action_cache=pauli_action_cache,
    )


def _apply_compiled_polynomial(state: np.ndarray, compiled_poly: CompiledPolynomialAction) -> np.ndarray:
    """Apply a compiled PauliPolynomial action to a statevector."""
    if int(getattr(compiled_poly, "nq", 0)) == 0 and len(compiled_poly.terms) == 0:
        return np.zeros_like(state)
    return _apply_compiled_polynomial_shared(state, compiled_poly)


def _apply_exp_term(
    psi: np.ndarray, action: CompiledPauliAction, coeff: complex, alpha: float, tol: float = 1e-12,
) -> np.ndarray:
    return _apply_exp_term_shared(
        psi=psi,
        action=action,
        coeff=complex(coeff),
        dt=float(alpha),
        tol=float(tol),
    )


def _evolve_trotter_suzuki2_absolute(
    psi0, ordered_labels, coeff_map, compiled_actions, time_value, trotter_steps,
) -> np.ndarray:
    psi = np.array(psi0, copy=True)
    if abs(time_value) <= 1e-15:
        return psi
    dt = float(time_value) / float(trotter_steps)
    half = 0.5 * dt
    for _ in range(trotter_steps):
        for label in ordered_labels:
            psi = _apply_exp_term(psi, compiled_actions[label], coeff_map[label], half)
        for label in reversed(ordered_labels):
            psi = _apply_exp_term(psi, compiled_actions[label], coeff_map[label], half)
    return _normalize_state(psi)


def _expectation_hamiltonian(psi: np.ndarray, hmat: np.ndarray) -> float:
    return float(np.real(np.vdot(psi, hmat @ psi)))


# ---------------------------------------------------------------------------
# Observables (identical to VQE pipeline)
# ---------------------------------------------------------------------------

def _occupation_site0(psi: np.ndarray, num_sites: int) -> tuple[float, float]:
    probs = np.abs(psi) ** 2
    n_up = 0.0
    n_dn = 0.0
    for idx, prob in enumerate(probs):
        n_up += float((idx >> 0) & 1) * float(prob)
        n_dn += float((idx >> num_sites) & 1) * float(prob)
    return float(n_up), float(n_dn)


def _doublon_total(psi: np.ndarray, num_sites: int) -> float:
    probs = np.abs(psi) ** 2
    out = 0.0
    for idx, prob in enumerate(probs):
        count = 0
        for site in range(num_sites):
            up = (idx >> site) & 1
            dn = (idx >> (num_sites + site)) & 1
            count += int(up * dn)
        out += float(count) * float(prob)
    return float(out)


def _state_to_amplitudes_qn_to_q0(psi: np.ndarray, cutoff: float = 1e-12) -> dict[str, dict[str, float]]:
    nq = int(round(math.log2(psi.size)))
    out: dict[str, dict[str, float]] = {}
    for idx, amp in enumerate(psi):
        if abs(amp) < cutoff:
            continue
        bit = format(idx, f"0{nq}b")
        out[bit] = {"re": float(np.real(amp)), "im": float(np.imag(amp))}
    return out


def _state_from_amplitudes_qn_to_q0(
    amplitudes_qn_to_q0: dict[str, Any],
    nq_total: int,
) -> np.ndarray:
    if not isinstance(amplitudes_qn_to_q0, dict) or len(amplitudes_qn_to_q0) == 0:
        raise ValueError("Missing or empty initial_state.amplitudes_qn_to_q0 in ADAPT JSON.")
    dim = 1 << int(nq_total)
    psi = np.zeros(dim, dtype=complex)
    for bitstr, comp in amplitudes_qn_to_q0.items():
        if not isinstance(bitstr, str) or len(bitstr) != int(nq_total) or any(ch not in "01" for ch in bitstr):
            raise ValueError(f"Invalid bitstring key in ADAPT amplitudes: {bitstr!r}")
        if not isinstance(comp, dict):
            raise ValueError(f"Amplitude payload for bitstring {bitstr!r} must be a dict.")
        re_val = float(comp.get("re", 0.0))
        im_val = float(comp.get("im", 0.0))
        idx = int(bitstr, 2)
        psi[idx] = complex(re_val, im_val)
    return _normalize_state(psi)


def _load_adapt_initial_state(
    adapt_json_path: Path,
    nq_total: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not adapt_json_path.exists():
        raise FileNotFoundError(f"ADAPT input JSON not found: {adapt_json_path}")
    raw = json.loads(adapt_json_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("ADAPT input JSON must be a top-level object.")
    initial_state = raw.get("initial_state")
    if not isinstance(initial_state, dict):
        raise ValueError("ADAPT input JSON missing object key: initial_state")
    amplitudes = initial_state.get("amplitudes_qn_to_q0")
    psi = _state_from_amplitudes_qn_to_q0(amplitudes, int(nq_total))
    meta = {
        "settings": raw.get("settings", {}),
        "adapt_vqe": raw.get("adapt_vqe", {}),
        "ground_state": raw.get("ground_state", {}),
        "vqe": raw.get("vqe", {}),
        "initial_state_source": initial_state.get("source"),
    }
    return psi, meta


_HH_STAGED_CONTINUATION_MODES = frozenset({"phase1_v1", "phase2_v1", "phase3_v1"})


def _extract_nested(payload: Mapping[str, Any], *keys: str) -> Any:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _resolve_exact_energy_from_payload(payload: Mapping[str, Any]) -> float | None:
    candidates = (
        _extract_nested(payload, "ground_state", "exact_energy_filtered"),
        _extract_nested(payload, "ground_state", "exact_energy"),
        _extract_nested(payload, "adapt_vqe", "exact_gs_energy"),
        _extract_nested(payload, "vqe", "exact_energy"),
    )
    for raw in candidates:
        if raw is None:
            continue
        try:
            value = float(raw)
        except Exception:
            continue
        if np.isfinite(value):
            return float(value)
    return None


def _validate_adapt_ref_metadata_for_exact_reuse(
    *,
    adapt_settings: Mapping[str, Any],
    args: argparse.Namespace,
    is_hh: bool,
    float_tol: float = 1e-10,
) -> list[str]:
    if not isinstance(adapt_settings, Mapping):
        return ["settings missing from adapt_ref_json"]

    mismatches: list[str] = []

    def _cmp_scalar(field: str, expected: Any, actual: Any) -> None:
        if actual != expected:
            mismatches.append(f"{field}: expected={expected!r} adapt_ref_json={actual!r}")

    def _cmp_float(field: str, expected: float, actual_raw: Any) -> None:
        try:
            actual = float(actual_raw)
        except Exception:
            mismatches.append(f"{field}: expected={expected!r} adapt_ref_json={actual_raw!r}")
            return
        if abs(float(expected) - actual) > float(float_tol):
            mismatches.append(f"{field}: expected={float(expected)!r} adapt_ref_json={actual!r}")

    _cmp_scalar("L", int(args.L), adapt_settings.get("L"))
    _cmp_scalar("problem", str(args.problem).strip().lower(), str(adapt_settings.get("problem", "")).strip().lower())
    _cmp_scalar("ordering", str(args.ordering), adapt_settings.get("ordering"))
    _cmp_scalar("boundary", str(args.boundary), adapt_settings.get("boundary"))
    _cmp_float("t", float(args.t), adapt_settings.get("t"))
    _cmp_float("u", float(args.u), adapt_settings.get("u"))
    _cmp_float("dv", float(args.dv), adapt_settings.get("dv"))

    if bool(is_hh):
        _cmp_float("omega0", float(args.omega0), adapt_settings.get("omega0"))
        _cmp_float("g_ep", float(args.g_ep), adapt_settings.get("g_ep"))
        _cmp_scalar("n_ph_max", int(args.n_ph_max), adapt_settings.get("n_ph_max"))
        _cmp_scalar("boson_encoding", str(args.boson_encoding), adapt_settings.get("boson_encoding"))

    return mismatches


def _resolve_exact_energy_override_from_adapt_ref(
    *,
    adapt_ref_meta: Mapping[str, Any] | None,
    args: argparse.Namespace,
    problem: str,
    continuation_mode: str | None,
) -> tuple[float | None, str, list[str]]:
    if not isinstance(adapt_ref_meta, Mapping):
        return None, "computed", []
    if str(problem).strip().lower() != "hh":
        return None, "computed", []
    mode_key = str(continuation_mode if continuation_mode is not None else "legacy").strip().lower()
    if mode_key not in _HH_STAGED_CONTINUATION_MODES:
        return None, "computed", []

    mismatches = _validate_adapt_ref_metadata_for_exact_reuse(
        adapt_settings=adapt_ref_meta.get("settings", {}),
        args=args,
        is_hh=True,
    )
    if len(mismatches) > 0:
        return None, "computed", mismatches

    exact_energy = _resolve_exact_energy_from_payload(adapt_ref_meta)
    if exact_energy is None:
        return None, "computed", []
    return float(exact_energy), "adapt_ref_json", []


# ============================================================================
# ADAPT-VQE core — standard algorithm, no meta-learner
# ============================================================================

@dataclass
class AdaptVQEResult:
    """Result container for the hardcoded ADAPT-VQE run."""
    energy: float
    theta: np.ndarray
    selected_ops: list[AnsatzTerm]
    history: list[dict[str, Any]]
    stop_reason: str
    nfev_total: int


def _build_uccsd_pool(
    num_sites: int,
    num_particles: tuple[int, int],
    ordering: str,
) -> list[AnsatzTerm]:
    """Build the UCCSD operator pool using HardcodedUCCSDAnsatz.base_terms.

    This reuses the exact same excitation generators the VQE pipeline uses,
    ensuring apples-to-apples comparison with the Qiskit UCCSD pool.
    """
    dummy_ansatz = HardcodedUCCSDAnsatz(
        dims=int(num_sites),
        num_particles=num_particles,
        reps=1,
        repr_mode="JW",
        indexing=str(ordering),
        include_singles=True,
        include_doubles=True,
    )
    return list(dummy_ansatz.base_terms)


def _build_cse_pool(
    num_sites: int,
    ordering: str,
    t: float,
    u: float,
    dv: float,
    boundary: str,
) -> list[AnsatzTerm]:
    """Build a CSE-style pool from the term-wise Hubbard ansatz base terms."""
    dummy_ansatz = HubbardTermwiseAnsatz(
        dims=int(num_sites),
        t=float(t),
        U=float(u),
        v=float(dv),
        reps=1,
        repr_mode="JW",
        indexing=str(ordering),
        pbc=(str(boundary).strip().lower() == "periodic"),
        include_potential_terms=True,
    )
    return list(dummy_ansatz.base_terms)


def _build_full_hamiltonian_pool(
    h_poly: Any,
    tol: float = 1e-12,
    normalize_coeff: bool = False,
) -> list[AnsatzTerm]:
    """Build a pool with one generator per non-identity Hamiltonian Pauli term."""
    pool: list[AnsatzTerm] = []
    terms = h_poly.return_polynomial()
    if not terms:
        return pool
    nq = int(terms[0].nqubit())
    id_label = "e" * nq

    for term in terms:
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if label == id_label:
            continue
        if abs(coeff) <= tol:
            continue
        if abs(coeff.imag) > tol:
            raise ValueError(
                f"Non-negligible imaginary Hamiltonian coefficient for term {label}: {coeff}"
            )
        generator = PauliPolynomial("JW")
        term_coeff = 1.0 if bool(normalize_coeff) else float(coeff.real)
        label_prefix = "ham_unit_term" if bool(normalize_coeff) else "ham_term"
        generator.add_term(PauliTerm(nq, ps=label, pc=float(term_coeff)))
        pool.append(AnsatzTerm(label=f"{label_prefix}({label})", polynomial=generator))
    return pool


def _polynomial_signature(poly: Any, tol: float = 1e-12) -> tuple[tuple[str, float], ...]:
    """Canonical real-valued signature for deduplicating PauliPolynomial generators."""
    items: list[tuple[str, float]] = []
    for term in poly.return_polynomial():
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if abs(coeff) <= tol:
            continue
        if abs(coeff.imag) > tol:
            raise ValueError(f"Non-negligible imaginary coefficient in pool polynomial: {coeff} ({label})")
        items.append((label, round(float(coeff.real), 12)))
    items.sort()
    return tuple(items)


def _build_hh_termwise_augmented_pool(h_poly: Any, tol: float = 1e-12) -> list[AnsatzTerm]:
    """HH-only termwise pool: unit-normalized Hamiltonian terms + x->y quadrature partners."""
    base_pool = _build_full_hamiltonian_pool(h_poly, tol=tol, normalize_coeff=True)
    if not base_pool:
        return []

    terms = h_poly.return_polynomial()
    nq = int(terms[0].nqubit())
    id_label = "e" * nq

    seen_labels: set[str] = set()
    for op in base_pool:
        op_terms = op.polynomial.return_polynomial()
        if not op_terms:
            continue
        seen_labels.add(str(op_terms[0].pw2strng()))

    aug_pool = list(base_pool)
    for term in terms:
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if label == id_label or abs(coeff) <= tol:
            continue
        if "x" not in label:
            continue
        y_label = label.replace("x", "y")
        if y_label in seen_labels:
            continue
        gen = PauliPolynomial("JW")
        # Keep quadrature partners physically scaled to avoid over-dominating early ADAPT steps.
        y_coeff = abs(float(coeff.real))
        if y_coeff <= tol:
            y_coeff = 1.0
        gen.add_term(PauliTerm(nq, ps=y_label, pc=y_coeff))
        aug_pool.append(AnsatzTerm(label=f"ham_quadrature_term({y_label})", polynomial=gen))
        seen_labels.add(y_label)
    return aug_pool


def _build_hva_pool(
    num_sites: int,
    t: float,
    u: float,
    omega0: float,
    g_ep: float,
    dv: float,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
) -> list[AnsatzTerm]:
    # 1) Preserve the original HH layerwise generators used by the HVA form.
    layerwise = HubbardHolsteinLayerwiseAnsatz(
        dims=int(num_sites),
        J=float(t),
        U=float(u),
        omega0=float(omega0),
        g=float(g_ep),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        v=None,
        v_t=None,
        v0=float(dv),
        t_eval=None,
        reps=1,
        repr_mode="JW",
        indexing=str(ordering),
        pbc=(str(boundary).strip().lower() == "periodic"),
        include_zero_point=True,
    )
    pool: list[AnsatzTerm] = list(layerwise.base_terms)

    # 2) Augment with fermionic UCCSD-style generators lifted to HH register.
    # This keeps the electron-number-conserving structure while allowing the
    # ADAPT search to explore a richer, HH-relevant operator manifold.
    n_sites = int(num_sites)
    boson_bits = n_sites * int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
    hf_num_particles = half_filled_num_particles(n_sites)
    uccsd_kwargs = {
        "dims": n_sites,
        "num_particles": hf_num_particles,
        "include_singles": True,
        "include_doubles": True,
        "repr_mode": "JW",
        "indexing": str(ordering),
    }
    if str(boundary).strip().lower() == "periodic":
        try:
            uccsd_kwargs["pbc"] = True
            uccsd = HardcodedUCCSDAnsatz(**uccsd_kwargs)
        except TypeError as exc:
            if "pbc" not in str(exc):
                raise
            uccsd_kwargs.pop("pbc", None)
            uccsd = HardcodedUCCSDAnsatz(**uccsd_kwargs)
    else:
        uccsd = HardcodedUCCSDAnsatz(**uccsd_kwargs)
    for t_i in uccsd.base_terms:
        for term_idx, term in enumerate(t_i.polynomial.return_polynomial()):
            coeff = complex(term.p_coeff)
            if abs(coeff) <= 1e-15:
                continue
            if abs(coeff.imag) > 1e-12:
                raise ValueError(
                    f"Non-negligible imaginary UCCSD coefficient for HH pool term {t_i.label}: {coeff}"
                )
            base = str(term.pw2strng())
            padded = ("e" * boson_bits) + base
            lifted = PauliPolynomial("JW")
            lifted.add_term(PauliTerm(2 * n_sites + boson_bits, ps=padded, pc=float(coeff.real)))
            pool.append(AnsatzTerm(label=f"{t_i.label}_{term_idx}", polynomial=lifted))

    return pool


def _build_hh_uccsd_fermion_lifted_pool(
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
    num_particles: tuple[int, int] | None = None,
) -> list[AnsatzTerm]:
    """HH-only UCCSD pool lifted into full HH register with boson identity prefix."""
    n_sites = int(num_sites)
    num_particles_eff = tuple(num_particles) if num_particles is not None else tuple(half_filled_num_particles(n_sites))
    ferm_nq = 2 * n_sites
    boson_bits = n_sites * int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
    nq_total = ferm_nq + boson_bits

    uccsd_kwargs = {
        "dims": n_sites,
        "num_particles": num_particles_eff,
        "include_singles": True,
        "include_doubles": True,
        "repr_mode": "JW",
        "indexing": str(ordering),
    }
    if str(boundary).strip().lower() == "periodic":
        try:
            uccsd_kwargs["pbc"] = True
            uccsd = HardcodedUCCSDAnsatz(**uccsd_kwargs)
        except TypeError as exc:
            if "pbc" not in str(exc):
                raise
            uccsd_kwargs.pop("pbc", None)
            uccsd = HardcodedUCCSDAnsatz(**uccsd_kwargs)
    else:
        uccsd = HardcodedUCCSDAnsatz(**uccsd_kwargs)

    lifted_pool: list[AnsatzTerm] = []
    for op in uccsd.base_terms:
        lifted = PauliPolynomial("JW")
        for term in op.polynomial.return_polynomial():
            coeff = complex(term.p_coeff)
            if abs(coeff) <= 1e-15:
                continue
            if abs(coeff.imag) > 1e-12:
                raise ValueError(f"Non-negligible imaginary UCCSD coefficient in {op.label}: {coeff}")
            ferm_ps = str(term.pw2strng())
            if len(ferm_ps) != ferm_nq:
                raise ValueError(
                    f"Unexpected fermion Pauli length {len(ferm_ps)} != {ferm_nq} for UCCSD operator {op.label}"
                )
            full_ps = ("e" * boson_bits) + ferm_ps
            lifted.add_term(PauliTerm(nq_total, ps=full_ps, pc=float(coeff.real)))
        if len(lifted.return_polynomial()) == 0:
            continue
        lifted_pool.append(AnsatzTerm(label=f"uccsd_ferm_lifted::{op.label}", polynomial=lifted))
    return lifted_pool


def _build_paop_pool(
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
    pool_key: str,
    paop_r: int,
    paop_split_paulis: bool,
    paop_prune_eps: float,
    paop_normalization: str,
    num_particles: tuple[int, int],
) -> list[AnsatzTerm]:
    if make_paop_pool is None:
        raise RuntimeError(f"PAOP pool requested but operator_pools module unavailable: {_PAOP_IMPORT_ERROR}")

    pool_specs = make_paop_pool(
        pool_key,
        num_sites=int(num_sites),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        ordering=str(ordering),
        boundary=str(boundary),
        paop_r=int(paop_r),
        paop_split_paulis=bool(paop_split_paulis),
        paop_prune_eps=float(paop_prune_eps),
        paop_normalization=str(paop_normalization),
        num_particles=tuple(num_particles),
    )
    return [AnsatzTerm(label=label, polynomial=poly) for label, poly in pool_specs]


def _deduplicate_pool_terms(pool: list[AnsatzTerm]) -> list[AnsatzTerm]:
    """Deduplicate pool operators by canonical polynomial signature."""
    seen: set[tuple[tuple[str, float], ...]] = set()
    dedup_pool: list[AnsatzTerm] = []
    for term in pool:
        sig = _polynomial_signature(term.polynomial)
        if sig in seen:
            continue
        seen.add(sig)
        dedup_pool.append(term)
    return dedup_pool


def _polynomial_signature_digest(poly: Any, tol: float = 1e-12) -> str:
    """Low-memory ordered polynomial signature digest."""
    h = hashlib.sha1()
    for term in poly.return_polynomial():
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tol):
            continue
        if abs(coeff.imag) > 1e-10:
            raise ValueError(f"Non-negligible imaginary coefficient in pool term: {coeff}")
        label = str(term.pw2strng())
        coeff_real = round(float(coeff.real), 12)
        h.update(label.encode("ascii", errors="ignore"))
        h.update(b":")
        h.update(f"{coeff_real:+.12e}".encode("ascii"))
        h.update(b";")
    return h.hexdigest()


def _deduplicate_pool_terms_lightweight(pool: list[AnsatzTerm]) -> list[AnsatzTerm]:
    """Deduplicate pool operators with a streaming digest to reduce peak memory."""
    seen: set[str] = set()
    dedup_pool: list[AnsatzTerm] = []
    for term in pool:
        sig = _polynomial_signature_digest(term.polynomial)
        if sig in seen:
            continue
        seen.add(sig)
        dedup_pool.append(term)
    return dedup_pool


def _build_hh_full_meta_pool(
    *,
    h_poly: Any,
    num_sites: int,
    t: float,
    u: float,
    omega0: float,
    g_ep: float,
    dv: float,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
    paop_r: int,
    paop_split_paulis: bool,
    paop_prune_eps: float,
    paop_normalization: str,
    num_particles: tuple[int, int],
) -> tuple[list[AnsatzTerm], dict[str, int]]:
    """Build HH full meta-pool: uccsd_lifted + hva + paop_full + paop_lf_full."""
    uccsd_lifted_pool = _build_hh_uccsd_fermion_lifted_pool(
        int(num_sites),
        int(n_ph_max),
        str(boson_encoding),
        str(ordering),
        str(boundary),
        num_particles=num_particles,
    )
    hva_pool = _build_hva_pool(
        int(num_sites),
        float(t),
        float(u),
        float(omega0),
        float(g_ep),
        float(dv),
        int(n_ph_max),
        str(boson_encoding),
        str(ordering),
        str(boundary),
    )
    termwise_aug: list[AnsatzTerm] = []
    if abs(float(g_ep)) > 1e-15:
        termwise_aug = [
            AnsatzTerm(label=f"hh_termwise_{term.label}", polynomial=term.polynomial)
            for term in _build_hh_termwise_augmented_pool(h_poly)
        ]
    paop_full_pool = _build_paop_pool(
        int(num_sites),
        int(n_ph_max),
        str(boson_encoding),
        str(ordering),
        str(boundary),
        "paop_full",
        int(paop_r),
        bool(paop_split_paulis),
        float(paop_prune_eps),
        str(paop_normalization),
        num_particles,
    )
    paop_lf_full_pool = _build_paop_pool(
        int(num_sites),
        int(n_ph_max),
        str(boson_encoding),
        str(ordering),
        str(boundary),
        "paop_lf_full",
        int(paop_r),
        bool(paop_split_paulis),
        float(paop_prune_eps),
        str(paop_normalization),
        num_particles,
    )
    merged = (
        list(uccsd_lifted_pool)
        + list(hva_pool)
        + list(termwise_aug)
        + list(paop_full_pool)
        + list(paop_lf_full_pool)
    )
    meta = {
        "raw_uccsd_lifted": int(len(uccsd_lifted_pool)),
        "raw_hva": int(len(hva_pool)),
        "raw_hh_termwise_augmented": int(len(termwise_aug)),
        "raw_paop_full": int(len(paop_full_pool)),
        "raw_paop_lf_full": int(len(paop_lf_full_pool)),
        "raw_total": int(len(merged)),
    }
    # n_ph_max>=2 can create very large term signatures; use streaming digest
    # dedup to avoid high transient memory spikes from tuple materialization.
    if int(n_ph_max) >= 2:
        dedup_pool = _deduplicate_pool_terms_lightweight(merged)
    else:
        dedup_pool = _deduplicate_pool_terms(merged)
    return dedup_pool, meta


def _apply_pauli_polynomial_uncached(state: np.ndarray, poly: Any) -> np.ndarray:
    r"""Compute G|psi> where G is a PauliPolynomial (sum of weighted Pauli strings).

    G = \sum_j c_j P_j   =>   G|psi> = \sum_j c_j P_j|psi>
    """
    terms = poly.return_polynomial()
    if not terms:
        return np.zeros_like(state)
    nq = int(terms[0].nqubit())
    id_str = "e" * nq
    result = np.zeros_like(state)
    for term in terms:
        ps = term.pw2strng()
        coeff = complex(term.p_coeff)
        if abs(coeff) < 1e-15:
            continue
        if ps == id_str:
            result += coeff * state
        else:
            result += coeff * apply_pauli_string(state, ps)
    return result


def _apply_pauli_polynomial(
    state: np.ndarray,
    poly: Any,
    *,
    compiled: CompiledPolynomialAction | None = None,
) -> np.ndarray:
    if compiled is not None:
        return _apply_compiled_polynomial(state, compiled)
    return _apply_pauli_polynomial_uncached(state, poly)


def _commutator_gradient(
    h_poly: Any,
    pool_op: AnsatzTerm,
    psi_current: np.ndarray,
    *,
    h_compiled: CompiledPolynomialAction | None = None,
    pool_compiled: CompiledPolynomialAction | None = None,
    hpsi_precomputed: np.ndarray | None = None,
) -> float:
    r"""Compute dE/dtheta at theta=0 for appending pool_op to the current state.

    E(theta) = <psi|exp(+i theta G) H exp(-i theta G)|psi>

    The analytic gradient at theta=0 is:
        dE/dtheta|_0 = i <psi|[H, G]|psi> = 2 Im(<psi|H G|psi>)

    Since H is Hermitian: <psi|H G|psi> = <H psi | G psi>.

    This is exact and works for multi-term PauliPolynomial generators
    (unlike the parameter-shift rule which requires single-Pauli generators).
    """
    g_psi = _apply_pauli_polynomial(psi_current, pool_op.polynomial, compiled=pool_compiled)
    h_psi = (
        np.asarray(hpsi_precomputed, dtype=complex)
        if hpsi_precomputed is not None
        else _apply_pauli_polynomial(psi_current, h_poly, compiled=h_compiled)
    )
    return adapt_commutator_grad_from_hpsi(h_psi, g_psi)


def _prepare_adapt_state(
    psi_ref: np.ndarray,
    selected_ops: list[AnsatzTerm],
    theta: np.ndarray,
) -> np.ndarray:
    """Apply the current ADAPT ansatz: prod_k exp(-i theta_k G_k) |ref>."""
    psi = np.array(psi_ref, copy=True)
    for k, op in enumerate(selected_ops):
        psi = apply_exp_pauli_polynomial(psi, op.polynomial, float(theta[k]))
    return psi


def _adapt_energy_fn(
    h_poly: Any,
    psi_ref: np.ndarray,
    selected_ops: list[AnsatzTerm],
    theta: np.ndarray,
    *,
    h_compiled: CompiledPolynomialAction | None = None,
) -> float:
    """Energy of the current ADAPT ansatz at parameters theta."""
    psi = _prepare_adapt_state(psi_ref, selected_ops, theta)
    if h_compiled is not None:
        energy, _hpsi = energy_via_one_apply(psi, h_compiled)
        return float(energy)
    return float(expval_pauli_polynomial(psi, h_poly))


def _exact_gs_energy_for_problem(
    h_poly: Any,
    *,
    problem: str,
    num_sites: int,
    num_particles: tuple[int, int],
    indexing: str,
    n_ph_max: int = 1,
    boson_encoding: str = "binary",
    t: float | None = None,
    u: float | None = None,
    dv: float | None = None,
    omega0: float | None = None,
    g_ep: float | None = None,
    boundary: str = "open",
) -> float:
    """Dispatch to the correct sector-filtered exact ground energy.

    For problem='hh', use fermion-only sector filtering (phonon qubits free).
    For problem='hubbard', use standard full-register sector filtering.
    """
    if str(problem).strip().lower() == "hh":
        if (
            t is not None
            and u is not None
            and dv is not None
            and omega0 is not None
            and g_ep is not None
        ):
            try:
                from src.quantum.ed_hubbard_holstein import build_hh_sector_hamiltonian_ed

                h_sector = build_hh_sector_hamiltonian_ed(
                    dims=int(num_sites),
                    J=float(t),
                    U=float(u),
                    omega0=float(omega0),
                    g=float(g_ep),
                    n_ph_max=int(n_ph_max),
                    num_particles=tuple(num_particles),
                    indexing=str(indexing),
                    boson_encoding=str(boson_encoding),
                    pbc=(str(boundary).strip().lower() == "periodic"),
                    delta_v=float(dv),
                    include_zero_point=True,
                    sparse=True,
                    return_basis=False,
                )
                try:
                    from scipy.sparse import spmatrix as _spmatrix
                    from scipy.sparse.linalg import eigsh as _eigsh

                    if isinstance(h_sector, _spmatrix):
                        eval0 = _eigsh(
                            h_sector,
                            k=1,
                            which="SA",
                            return_eigenvectors=False,
                            tol=1e-10,
                            maxiter=max(1000, 10 * int(h_sector.shape[0])),
                        )
                        return float(np.real(eval0[0]))
                except Exception:
                    pass

                h_dense = np.asarray(
                    h_sector.toarray() if hasattr(h_sector, "toarray") else h_sector,
                    dtype=complex,
                )
                evals = np.linalg.eigvalsh(h_dense)
                return float(np.min(np.real(evals)))
            except Exception as exc:
                _ai_log(
                    "hardcoded_adapt_hh_exact_sparse_fallback",
                    status="failed",
                    error=str(exc),
                )
        return exact_ground_energy_sector_hh(
            h_poly,
            num_sites=int(num_sites),
            num_particles=num_particles,
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            indexing=str(indexing),
        )
    else:
        return exact_ground_energy_sector(
            h_poly,
            num_sites=int(num_sites),
            num_particles=num_particles,
            indexing=str(indexing),
        )


def _exact_reference_state_for_hh(
    *,
    num_sites: int,
    num_particles: tuple[int, int],
    indexing: str,
    n_ph_max: int,
    boson_encoding: str,
    t: float,
    u: float,
    dv: float,
    omega0: float,
    g_ep: float,
    boundary: str,
) -> np.ndarray | None:
    try:
        from src.quantum.ed_hubbard_holstein import build_hh_sector_hamiltonian_ed
        from scipy.sparse import spmatrix as _spmatrix
        from scipy.sparse.linalg import eigsh as _eigsh

        h_sector, basis = build_hh_sector_hamiltonian_ed(
            dims=int(num_sites),
            J=float(t),
            U=float(u),
            omega0=float(omega0),
            g=float(g_ep),
            n_ph_max=int(n_ph_max),
            num_particles=tuple(num_particles),
            indexing=str(indexing),
            boson_encoding=str(boson_encoding),
            pbc=(str(boundary).strip().lower() == "periodic"),
            delta_v=float(dv),
            include_zero_point=True,
            sparse=True,
            return_basis=True,
        )
        if isinstance(h_sector, _spmatrix):
            evals, evecs = _eigsh(
                h_sector,
                k=1,
                which="SA",
                return_eigenvectors=True,
                tol=1e-10,
                maxiter=max(1000, 10 * int(h_sector.shape[0])),
            )
            vec_sector = np.asarray(evecs[:, 0], dtype=complex).reshape(-1)
        else:
            dense = np.asarray(h_sector, dtype=complex)
            evals, evecs = np.linalg.eigh(dense)
            vec_sector = np.asarray(evecs[:, int(np.argmin(np.real(evals)))], dtype=complex).reshape(-1)
        psi_full = np.zeros(1 << int(basis.total_qubits), dtype=complex)
        for local_idx, basis_idx in enumerate(basis.basis_indices):
            psi_full[int(basis_idx)] = complex(vec_sector[int(local_idx)])
        return _normalize_state(psi_full)
    except Exception as exc:
        _ai_log(
            "hardcoded_adapt_exact_reference_state_unavailable",
            error=str(exc),
        )
        return None


# ---------------------------------------------------------------------------
# Windowed reopt helpers (pure, deterministic)
# ---------------------------------------------------------------------------

_VALID_REOPT_POLICIES = {"append_only", "full", "windowed"}


def _resolve_reopt_active_indices(
    *,
    policy: str,
    n: int,
    theta: np.ndarray,
    window_size: int = 3,
    window_topk: int = 0,
    periodic_full_refit_triggered: bool = False,
) -> tuple[list[int], str]:
    """Return (sorted_active_indices, effective_policy_name).

    Active-index selection contract (windowed):
      1. w_eff = min(window_size, n)
      2. newest = [n - w_eff, ..., n - 1]
      3. older  = [0, ..., n - w_eff - 1]
      4. If window_topk > 0, rank older by descending |theta[i]|,
         break ties by ascending index i.
      5. k_eff = min(window_topk, len(older))
      6. active = union(newest, top-k older)
      7. return sorted ascending

    For append_only: active = [n - 1]
    For full or periodic full-refit override: active = [0 .. n-1]
    """
    policy_key = str(policy).strip().lower()
    if n <= 0:
        return [], policy_key

    if policy_key == "append_only":
        return [n - 1], "append_only"

    if policy_key == "full":
        return list(range(n)), "full"

    if policy_key != "windowed":
        raise ValueError(f"Unknown reopt policy '{policy_key}'.")

    # Periodic full-refit override for this depth
    if periodic_full_refit_triggered:
        return list(range(n)), "windowed_periodic_full"

    w_eff = min(int(window_size), n)
    newest = list(range(n - w_eff, n))

    older_start = n - w_eff
    if older_start <= 0 or int(window_topk) <= 0:
        return sorted(newest), "windowed"

    older_candidates = list(range(0, older_start))
    # Rank by descending |theta[i]|, tie-break ascending index i
    older_ranked = sorted(
        older_candidates,
        key=lambda i: (-abs(float(theta[i])), i),
    )
    k_eff = min(int(window_topk), len(older_ranked))
    selected_older = older_ranked[:k_eff]
    active = sorted(set(newest) | set(selected_older))
    return active, "windowed"


def _make_reduced_objective(
    full_theta: np.ndarray,
    active_indices: list[int],
    obj_fn: Any,
) -> tuple[Any, np.ndarray]:
    """Build a reduced-variable objective and its initial point.

    Returns (reduced_obj, x0_reduced) where:
      - reduced_obj(x_active) reconstructs a full theta from frozen+active
        and calls obj_fn(full_theta)
      - x0_reduced = full_theta[active_indices]
    """
    frozen_theta = np.array(full_theta, copy=True)
    active_idx = list(active_indices)
    n_active = len(active_idx)
    x0 = np.array([float(frozen_theta[i]) for i in active_idx], dtype=float)

    if n_active == len(frozen_theta):
        # Full prefix — no wrapping needed
        return obj_fn, np.array(frozen_theta, copy=True)

    def _reduced(x_active: np.ndarray) -> float:
        full = np.array(frozen_theta, copy=True)
        x_arr = np.asarray(x_active, dtype=float).ravel()
        for k, idx in enumerate(active_idx):
            full[idx] = float(x_arr[k])
        return float(obj_fn(full))

    return _reduced, x0


def _resolve_adapt_continuation_mode(*, problem: str, requested_mode: str | None) -> str:
    mode_raw = "legacy" if requested_mode is None else str(requested_mode).strip().lower()
    if mode_raw == "":
        return "legacy"
    if mode_raw not in {"legacy", "phase1_v1", "phase2_v1", "phase3_v1"}:
        raise ValueError("adapt_continuation_mode must be one of {'legacy','phase1_v1','phase2_v1','phase3_v1'}.")
    return str(mode_raw)


@dataclass(frozen=True)
class ResolvedAdaptStopPolicy:
    adapt_drop_floor: float
    adapt_drop_patience: int
    adapt_drop_min_depth: int
    adapt_grad_floor: float
    adapt_drop_floor_source: str
    adapt_drop_patience_source: str
    adapt_drop_min_depth_source: str
    adapt_grad_floor_source: str
    drop_policy_enabled: bool
    drop_policy_source: str
    eps_energy_termination_enabled: bool
    eps_grad_termination_enabled: bool


def _resolve_adapt_stop_policy(
    *,
    problem: str,
    continuation_mode: str,
    adapt_drop_floor: float | None,
    adapt_drop_patience: int | None,
    adapt_drop_min_depth: int | None,
    adapt_grad_floor: float | None,
) -> ResolvedAdaptStopPolicy:
    staged_hh = bool(
        str(problem).strip().lower() == "hh"
        and str(continuation_mode).strip().lower() in _HH_STAGED_CONTINUATION_MODES
    )

    def _resolve_float(raw: float | None, *, staged_value: float, default_value: float) -> tuple[float, str]:
        if raw is None:
            if staged_hh:
                return float(staged_value), "auto_hh_staged"
            return float(default_value), "default_off"
        return float(raw), "explicit"

    def _resolve_int(raw: int | None, *, staged_value: int, default_value: int) -> tuple[int, str]:
        if raw is None:
            if staged_hh:
                return int(staged_value), "auto_hh_staged"
            return int(default_value), "default_off"
        return int(raw), "explicit"

    drop_floor_resolved, drop_floor_source = _resolve_float(
        adapt_drop_floor,
        staged_value=5e-4,
        default_value=-1.0,
    )
    drop_patience_resolved, drop_patience_source = _resolve_int(
        adapt_drop_patience,
        staged_value=3,
        default_value=0,
    )
    drop_min_depth_resolved, drop_min_depth_source = _resolve_int(
        adapt_drop_min_depth,
        staged_value=12,
        default_value=0,
    )
    grad_floor_resolved, grad_floor_source = _resolve_float(
        adapt_grad_floor,
        staged_value=2e-2,
        default_value=-1.0,
    )
    drop_policy_enabled = bool(drop_floor_resolved >= 0.0 and drop_patience_resolved > 0)
    if staged_hh and all(src == "auto_hh_staged" for src in (
        drop_floor_source,
        drop_patience_source,
        drop_min_depth_source,
        grad_floor_source,
    )):
        drop_policy_source = "auto_hh_staged"
    elif any(src == "explicit" for src in (
        drop_floor_source,
        drop_patience_source,
        drop_min_depth_source,
        grad_floor_source,
    )):
        drop_policy_source = "explicit"
    else:
        drop_policy_source = "default_off"

    return ResolvedAdaptStopPolicy(
        adapt_drop_floor=float(drop_floor_resolved),
        adapt_drop_patience=int(drop_patience_resolved),
        adapt_drop_min_depth=int(drop_min_depth_resolved),
        adapt_grad_floor=float(grad_floor_resolved),
        adapt_drop_floor_source=str(drop_floor_source),
        adapt_drop_patience_source=str(drop_patience_source),
        adapt_drop_min_depth_source=str(drop_min_depth_source),
        adapt_grad_floor_source=str(grad_floor_source),
        drop_policy_enabled=bool(drop_policy_enabled),
        drop_policy_source=str(drop_policy_source),
        eps_energy_termination_enabled=(not staged_hh),
        eps_grad_termination_enabled=(not staged_hh),
    )


def _phase1_repeated_family_flat(
    *,
    history: list[dict[str, Any]],
    candidate_family: str,
    patience: int,
    weak_drop_threshold: float,
) -> bool:
    if str(candidate_family).strip() == "":
        return False
    tail = [row for row in history if isinstance(row, dict) and row.get("candidate_family") is not None]
    need = max(0, int(patience) - 1)
    if need <= 0:
        return False
    if len(tail) < need:
        return False
    recent = tail[-need:]
    for row in recent:
        if str(row.get("candidate_family")) != str(candidate_family):
            return False
        drop = float(row.get("delta_abs_drop_from_prev", float("inf")))
        if not math.isfinite(drop) or drop > float(weak_drop_threshold):
            return False
    return True


def _splice_candidate_at_position(
    *,
    ops: list[AnsatzTerm],
    theta: np.ndarray,
    op: AnsatzTerm,
    position_id: int,
    init_theta: float = 0.0,
) -> tuple[list[AnsatzTerm], np.ndarray]:
    append_position = int(theta.size)
    pos = max(0, min(int(append_position), int(position_id)))
    new_ops = list(ops)
    new_ops.insert(pos, op)
    theta_arr = np.asarray(theta, dtype=float).reshape(-1)
    new_theta = np.insert(theta_arr, pos, float(init_theta))
    return new_ops, np.asarray(new_theta, dtype=float)


def _predict_reopt_window_for_position(
    *,
    theta: np.ndarray,
    position_id: int,
    policy: str,
    window_size: int,
    window_topk: int,
    periodic_full_refit_triggered: bool,
) -> list[int]:
    theta_arr = np.asarray(theta, dtype=float).reshape(-1)
    append_position = int(theta_arr.size)
    pos = max(0, min(int(append_position), int(position_id)))
    theta_hyp = np.insert(theta_arr, pos, 0.0)
    active, _name = _resolve_reopt_active_indices(
        policy=str(policy),
        n=int(theta_hyp.size),
        theta=np.asarray(theta_hyp, dtype=float),
        window_size=int(window_size),
        window_topk=int(window_topk),
        periodic_full_refit_triggered=bool(periodic_full_refit_triggered),
    )
    return [int(i) for i in active]


def _window_terms_for_position(
    *,
    selected_ops: list[AnsatzTerm],
    refit_window_indices: list[int],
    position_id: int,
) -> tuple[list[AnsatzTerm], list[str]]:
    window_terms: list[AnsatzTerm] = []
    window_labels: list[str] = []
    pos = int(position_id)
    for idx in refit_window_indices:
        j = int(idx)
        if j == pos:
            continue
        mapped = j if j < pos else j - 1
        if 0 <= int(mapped) < len(selected_ops):
            term = selected_ops[int(mapped)]
            window_terms.append(term)
            window_labels.append(str(term.label))
    return window_terms, window_labels


def _phase2_record_sort_key(record: Mapping[str, Any]) -> tuple[float, float, int, int]:
    return (
        -float(record.get("full_v2_score", float("-inf"))),
        -float(record.get("simple_score", float("-inf"))),
        int(record.get("candidate_pool_index", -1)),
        int(record.get("position_id", -1)),
    )


def _run_hardcoded_adapt_vqe(
    *,
    h_poly: Any,
    num_sites: int,
    ordering: str,
    problem: str,
    adapt_pool: str | None,
    t: float,
    u: float,
    dv: float,
    boundary: str,
    omega0: float,
    g_ep: float,
    n_ph_max: int,
    boson_encoding: str,
    max_depth: int,
    eps_grad: float,
    eps_energy: float,
    maxiter: int,
    seed: int,
    adapt_inner_optimizer: str = "SPSA",
    adapt_spsa_a: float = 0.2,
    adapt_spsa_c: float = 0.1,
    adapt_spsa_alpha: float = 0.602,
    adapt_spsa_gamma: float = 0.101,
    adapt_spsa_A: float = 10.0,
    adapt_spsa_avg_last: int = 0,
    adapt_spsa_eval_repeats: int = 1,
    adapt_spsa_eval_agg: str = "mean",
    adapt_spsa_callback_every: int = 1,
    adapt_spsa_progress_every_s: float = 60.0,
    allow_repeats: bool,
    finite_angle_fallback: bool,
    finite_angle: float,
    finite_angle_min_improvement: float,
    adapt_drop_floor: float | None = None,
    adapt_drop_patience: int | None = None,
    adapt_drop_min_depth: int | None = None,
    adapt_grad_floor: float | None = None,
    adapt_eps_energy_min_extra_depth: int = -1,
    adapt_eps_energy_patience: int = -1,
    adapt_ref_base_depth: int = 0,
    paop_r: int = 0,
    paop_split_paulis: bool = False,
    paop_prune_eps: float = 0.0,
    paop_normalization: str = "none",
    disable_hh_seed: bool = False,
    psi_ref_override: np.ndarray | None = None,
    adapt_gradient_parity_check: bool = False,
    adapt_state_backend: str = "compiled",
    adapt_reopt_policy: str = "append_only",
    adapt_window_size: int = 3,
    adapt_window_topk: int = 0,
    adapt_full_refit_every: int = 0,
    adapt_final_full_refit: bool = True,
    exact_gs_override: float | None = None,
    adapt_continuation_mode: str | None = "legacy",
    phase1_lambda_F: float = 1.0,
    phase1_lambda_compile: float = 0.05,
    phase1_lambda_measure: float = 0.02,
    phase1_lambda_leak: float = 0.0,
    phase1_score_z_alpha: float = 0.0,
    phase1_probe_max_positions: int = 6,
    phase1_plateau_patience: int = 2,
    phase1_trough_margin_ratio: float = 1.0,
    phase1_prune_enabled: bool = True,
    phase1_prune_fraction: float = 0.25,
    phase1_prune_max_candidates: int = 6,
    phase1_prune_max_regression: float = 1e-8,
    phase2_shortlist_fraction: float = 0.2,
    phase2_shortlist_size: int = 12,
    phase2_lambda_H: float = 1e-6,
    phase2_rho: float = 0.25,
    phase2_gamma_N: float = 1.0,
    phase2_enable_batching: bool = True,
    phase2_batch_target_size: int = 2,
    phase2_batch_size_cap: int = 3,
    phase2_batch_near_degenerate_ratio: float = 0.9,
    phase3_motif_source_json: Path | None = None,
    phase3_symmetry_mitigation_mode: str = "off",
    phase3_enable_rescue: bool = False,
    phase3_lifetime_cost_mode: str = "phase3_v1",
    phase3_runtime_split_mode: str = "off",
) -> tuple[dict[str, Any], np.ndarray]:
    """Run standard ADAPT-VQE and return (payload, psi_ground)."""
    if float(finite_angle) <= 0.0:
        raise ValueError("finite_angle must be > 0.")
    if float(finite_angle_min_improvement) < 0.0:
        raise ValueError("finite_angle_min_improvement must be >= 0.")
    adapt_state_backend_key = str(adapt_state_backend).strip().lower()
    if adapt_state_backend_key not in {"legacy", "compiled"}:
        raise ValueError("adapt_state_backend must be one of {'legacy','compiled'}.")
    adapt_reopt_policy_key = str(adapt_reopt_policy).strip().lower()
    if adapt_reopt_policy_key not in _VALID_REOPT_POLICIES:
        raise ValueError(f"adapt_reopt_policy must be one of {_VALID_REOPT_POLICIES}.")
    adapt_window_size_val = int(adapt_window_size)
    adapt_window_topk_val = int(adapt_window_topk)
    adapt_full_refit_every_val = int(adapt_full_refit_every)
    adapt_final_full_refit_val = bool(adapt_final_full_refit)
    if adapt_window_size_val < 1:
        raise ValueError("adapt_window_size must be >= 1.")
    if adapt_window_topk_val < 0:
        raise ValueError("adapt_window_topk must be >= 0.")
    if adapt_full_refit_every_val < 0:
        raise ValueError("adapt_full_refit_every must be >= 0.")
    adapt_inner_optimizer_key = str(adapt_inner_optimizer).strip().upper()
    if adapt_inner_optimizer_key not in {"COBYLA", "SPSA"}:
        raise ValueError("adapt_inner_optimizer must be one of {'COBYLA','SPSA'}.")
    adapt_spsa_eval_agg_key = str(adapt_spsa_eval_agg).strip().lower()
    if adapt_spsa_eval_agg_key not in {"mean", "median"}:
        raise ValueError("adapt_spsa_eval_agg must be one of {'mean','median'}.")
    if int(adapt_spsa_callback_every) < 1:
        raise ValueError("adapt_spsa_callback_every must be >= 1.")
    if float(adapt_spsa_progress_every_s) < 0.0:
        raise ValueError("adapt_spsa_progress_every_s must be >= 0.")
    if int(adapt_eps_energy_min_extra_depth) < -1:
        raise ValueError("adapt_eps_energy_min_extra_depth must be >= 0 or -1 (auto=L).")
    if int(adapt_eps_energy_patience) < -1 or int(adapt_eps_energy_patience) == 0:
        raise ValueError("adapt_eps_energy_patience must be >= 1 or -1 (auto=L).")
    if int(adapt_ref_base_depth) < 0:
        raise ValueError("adapt_ref_base_depth must be >= 0.")
    problem_key = str(problem).strip().lower()
    continuation_mode = _resolve_adapt_continuation_mode(
        problem=str(problem_key),
        requested_mode=adapt_continuation_mode,
    )
    stop_policy = _resolve_adapt_stop_policy(
        problem=str(problem_key),
        continuation_mode=str(continuation_mode),
        adapt_drop_floor=adapt_drop_floor,
        adapt_drop_patience=adapt_drop_patience,
        adapt_drop_min_depth=adapt_drop_min_depth,
        adapt_grad_floor=adapt_grad_floor,
    )
    adapt_drop_floor = float(stop_policy.adapt_drop_floor)
    adapt_drop_patience = int(stop_policy.adapt_drop_patience)
    adapt_drop_min_depth = int(stop_policy.adapt_drop_min_depth)
    adapt_grad_floor = float(stop_policy.adapt_grad_floor)
    drop_policy_enabled = bool(stop_policy.drop_policy_enabled)
    eps_energy_termination_enabled = bool(stop_policy.eps_energy_termination_enabled)
    eps_grad_termination_enabled = bool(stop_policy.eps_grad_termination_enabled)
    if float(adapt_drop_floor) >= 0.0 and int(adapt_drop_patience) < 1:
        raise ValueError("adapt_drop_patience must be >= 1 when adapt_drop_floor is enabled.")
    if float(adapt_drop_floor) >= 0.0 and int(adapt_drop_min_depth) < 1:
        raise ValueError("adapt_drop_min_depth must be >= 1 when adapt_drop_floor is enabled.")
    if int(adapt_drop_patience) < 0:
        raise ValueError("adapt_drop_patience must be >= 0.")
    if int(adapt_drop_min_depth) < 0:
        raise ValueError("adapt_drop_min_depth must be >= 0.")
    eps_energy_min_extra_depth_effective = (
        int(num_sites)
        if int(adapt_eps_energy_min_extra_depth) == -1
        else int(adapt_eps_energy_min_extra_depth)
    )
    eps_energy_patience_effective = (
        int(num_sites)
        if int(adapt_eps_energy_patience) == -1
        else int(adapt_eps_energy_patience)
    )
    if int(eps_energy_patience_effective) < 1:
        raise ValueError("resolved eps-energy patience must be >= 1.")
    if int(eps_energy_min_extra_depth_effective) < 0:
        raise ValueError("resolved eps-energy min extra depth must be >= 0.")
    phase3_symmetry_mitigation_mode_key = str(phase3_symmetry_mitigation_mode).strip().lower()
    if phase3_symmetry_mitigation_mode_key not in {"off", "verify_only", "postselect_diag_v1", "projector_renorm_v1"}:
        raise ValueError(
            "phase3_symmetry_mitigation_mode must be one of {'off','verify_only','postselect_diag_v1','projector_renorm_v1'}."
        )
    phase3_lifetime_cost_mode_key = str(phase3_lifetime_cost_mode).strip().lower()
    if phase3_lifetime_cost_mode_key not in {"off", "phase3_v1"}:
        raise ValueError("phase3_lifetime_cost_mode must be one of {'off','phase3_v1'}.")
    phase3_runtime_split_mode_key = str(phase3_runtime_split_mode).strip().lower()
    if phase3_runtime_split_mode_key not in {"off", "shortlist_pauli_children_v1"}:
        raise ValueError(
            "phase3_runtime_split_mode must be one of {'off','shortlist_pauli_children_v1'}."
        )
    pool_key_input = None if adapt_pool is None else str(adapt_pool).strip().lower()
    adapt_spsa_params = {
        "a": float(adapt_spsa_a),
        "c": float(adapt_spsa_c),
        "alpha": float(adapt_spsa_alpha),
        "gamma": float(adapt_spsa_gamma),
        "A": float(adapt_spsa_A),
        "avg_last": int(adapt_spsa_avg_last),
        "eval_repeats": int(adapt_spsa_eval_repeats),
        "eval_agg": str(adapt_spsa_eval_agg_key),
        "callback_every": int(adapt_spsa_callback_every),
        "progress_every_s": float(adapt_spsa_progress_every_s),
    }
    t0 = time.perf_counter()
    hf_bits = "N/A"
    _ai_log(
        "hardcoded_adapt_vqe_start",
        L=int(num_sites),
        problem=str(problem),
        adapt_pool=(str(pool_key_input) if pool_key_input is not None else None),
        adapt_continuation_mode=str(continuation_mode),
        phase3_motif_source_json=(str(phase3_motif_source_json) if phase3_motif_source_json is not None else None),
        phase3_symmetry_mitigation_mode=str(phase3_symmetry_mitigation_mode_key),
        phase3_runtime_split_mode=str(phase3_runtime_split_mode_key),
        phase3_enable_rescue=bool(phase3_enable_rescue),
        max_depth=int(max_depth),
        maxiter=int(maxiter),
        adapt_inner_optimizer=str(adapt_inner_optimizer_key),
        finite_angle_fallback=bool(finite_angle_fallback),
        finite_angle=float(finite_angle),
        finite_angle_min_improvement=float(finite_angle_min_improvement),
        adapt_gradient_parity_check=bool(adapt_gradient_parity_check),
        adapt_state_backend=str(adapt_state_backend_key),
        adapt_drop_policy_enabled=bool(drop_policy_enabled),
        adapt_drop_floor=(float(adapt_drop_floor) if drop_policy_enabled else None),
        adapt_drop_patience=(int(adapt_drop_patience) if drop_policy_enabled else None),
        adapt_drop_min_depth=(int(adapt_drop_min_depth) if drop_policy_enabled else None),
        adapt_grad_floor=(float(adapt_grad_floor) if float(adapt_grad_floor) >= 0.0 else None),
        adapt_drop_floor_resolved=float(adapt_drop_floor),
        adapt_drop_patience_resolved=int(adapt_drop_patience),
        adapt_drop_min_depth_resolved=int(adapt_drop_min_depth),
        adapt_grad_floor_resolved=float(adapt_grad_floor),
        adapt_drop_floor_source=str(stop_policy.adapt_drop_floor_source),
        adapt_drop_patience_source=str(stop_policy.adapt_drop_patience_source),
        adapt_drop_min_depth_source=str(stop_policy.adapt_drop_min_depth_source),
        adapt_grad_floor_source=str(stop_policy.adapt_grad_floor_source),
        adapt_drop_policy_source=str(stop_policy.drop_policy_source),
        adapt_eps_energy_min_extra_depth=int(adapt_eps_energy_min_extra_depth),
        adapt_eps_energy_patience=int(adapt_eps_energy_patience),
        adapt_ref_base_depth=int(adapt_ref_base_depth),
        adapt_eps_energy_min_extra_depth_effective=int(eps_energy_min_extra_depth_effective),
        adapt_eps_energy_patience_effective=int(eps_energy_patience_effective),
        eps_energy_gate_cumulative_depth=int(adapt_ref_base_depth) + int(eps_energy_min_extra_depth_effective),
        eps_energy_termination_enabled=bool(eps_energy_termination_enabled),
        eps_grad_termination_enabled=bool(eps_grad_termination_enabled),
        adapt_reopt_policy=str(adapt_reopt_policy_key),
        adapt_window_size=int(adapt_window_size_val),
        adapt_window_topk=int(adapt_window_topk_val),
        adapt_full_refit_every=int(adapt_full_refit_every_val),
        adapt_final_full_refit=bool(adapt_final_full_refit_val),
    )

    num_particles = half_filled_num_particles(int(num_sites))
    if problem_key == "hh":
        psi_ref = np.asarray(
            hubbard_holstein_reference_state(
                dims=int(num_sites),
                num_particles=num_particles,
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                indexing=str(ordering),
            ),
            dtype=complex,
        )
    else:
        hf_bits = str(hartree_fock_bitstring(
            n_sites=int(num_sites),
            num_particles=num_particles,
            indexing=str(ordering),
        ))
        nq = 2 * int(num_sites)
        psi_ref = np.asarray(basis_state(nq, hf_bits), dtype=complex)
    if psi_ref_override is not None:
        psi_ref_override_arr = np.asarray(psi_ref_override, dtype=complex).reshape(-1)
        if int(psi_ref_override_arr.size) != int(psi_ref.size):
            raise ValueError(
                f"psi_ref_override length mismatch: got {psi_ref_override_arr.size}, expected {psi_ref.size}"
            )
        psi_ref = _normalize_state(psi_ref_override_arr)
        _ai_log(
            "hardcoded_adapt_ref_override_applied",
            nq=int(round(math.log2(psi_ref.size))),
            dim=int(psi_ref.size),
        )

    # Build operator pool(s)
    def _build_hh_pool_by_key(pool_key_hh: str) -> tuple[list[AnsatzTerm], str]:
        key = str(pool_key_hh).strip().lower()
        if key == "hva":
            hva_pool = _build_hva_pool(
                int(num_sites),
                float(t),
                float(u),
                float(omega0),
                float(g_ep),
                float(dv),
                int(n_ph_max),
                str(boson_encoding),
                str(ordering),
                str(boundary),
            )
            if abs(float(g_ep)) <= 1e-15:
                return list(hva_pool), "hardcoded_adapt_vqe_hva_hh"
            ham_term_pool = _build_hh_termwise_augmented_pool(h_poly)
            merged_pool = list(hva_pool) + [
                AnsatzTerm(label=f"hh_termwise_{term.label}", polynomial=term.polynomial)
                for term in ham_term_pool
            ]
            seen: set[tuple[tuple[str, float], ...]] = set()
            dedup_pool: list[AnsatzTerm] = []
            for term in merged_pool:
                sig = _polynomial_signature(term.polynomial)
                if sig in seen:
                    continue
                seen.add(sig)
                dedup_pool.append(term)
            return dedup_pool, "hardcoded_adapt_vqe_hva_hh"
        if key == "full_meta":
            pool_full, full_meta_sizes = _build_hh_full_meta_pool(
                h_poly=h_poly,
                num_sites=int(num_sites),
                t=float(t),
                u=float(u),
                omega0=float(omega0),
                g_ep=float(g_ep),
                dv=float(dv),
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                ordering=str(ordering),
                boundary=str(boundary),
                paop_r=int(paop_r),
                paop_split_paulis=bool(paop_split_paulis),
                paop_prune_eps=float(paop_prune_eps),
                paop_normalization=str(paop_normalization),
                num_particles=num_particles,
            )
            _ai_log(
                "hardcoded_adapt_full_meta_pool_built",
                **full_meta_sizes,
                dedup_total=int(len(pool_full)),
            )
            return list(pool_full), "hardcoded_adapt_vqe_full_meta"
        if key == "uccsd_paop_lf_full":
            uccsd_lifted_pool = _build_hh_uccsd_fermion_lifted_pool(
                int(num_sites),
                int(n_ph_max),
                str(boson_encoding),
                str(ordering),
                str(boundary),
                num_particles=num_particles,
            )
            paop_pool = _build_paop_pool(
                int(num_sites),
                int(n_ph_max),
                str(boson_encoding),
                str(ordering),
                str(boundary),
                "paop_lf_full",
                int(paop_r),
                bool(paop_split_paulis),
                float(paop_prune_eps),
                str(paop_normalization),
                num_particles,
            )
            return _deduplicate_pool_terms(list(uccsd_lifted_pool) + list(paop_pool)), "hardcoded_adapt_vqe_uccsd_paop_lf_full"
        if key in {"paop", "paop_min", "paop_std", "paop_full", "paop_lf", "paop_lf_std", "paop_lf2_std", "paop_lf_full"}:
            paop_pool = _build_paop_pool(
                int(num_sites),
                int(n_ph_max),
                str(boson_encoding),
                str(ordering),
                str(boundary),
                key,
                int(paop_r),
                bool(paop_split_paulis),
                float(paop_prune_eps),
                str(paop_normalization),
                num_particles,
            )
            if abs(float(g_ep)) <= 1e-15:
                return list(paop_pool), f"hardcoded_adapt_vqe_{key}"
            hva_pool = _build_hva_pool(
                int(num_sites),
                float(t),
                float(u),
                float(omega0),
                float(g_ep),
                float(dv),
                int(n_ph_max),
                str(boson_encoding),
                str(ordering),
                str(boundary),
            )
            ham_term_pool = _build_hh_termwise_augmented_pool(h_poly)
            merged_pool = list(hva_pool) + [
                AnsatzTerm(label=f"hh_termwise_{term.label}", polynomial=term.polynomial)
                for term in ham_term_pool
            ] + list(paop_pool)
            seen: set[tuple[tuple[str, float], ...]] = set()
            dedup_pool: list[AnsatzTerm] = []
            for term in merged_pool:
                sig = _polynomial_signature(term.polynomial)
                if sig in seen:
                    continue
                seen.add(sig)
                dedup_pool.append(term)
            return dedup_pool, f"hardcoded_adapt_vqe_{key}"
        if key == "full_hamiltonian":
            return _build_full_hamiltonian_pool(h_poly, normalize_coeff=True), "hardcoded_adapt_vqe_full_hamiltonian_hh"
        raise ValueError(
            "For problem='hh', supported ADAPT pools are: "
            "hva, full_meta, uccsd_paop_lf_full, paop, paop_min, paop_std, paop_full, "
            "paop_lf, paop_lf_std, paop_lf2_std, paop_lf_full, full_hamiltonian"
        )

    pool_stage_family: list[str] = []
    pool_family_ids: list[str] = []
    phase1_core_limit = 0
    phase1_residual_indices: set[int] = set()
    if continuation_mode in {"phase1_v1", "phase2_v1", "phase3_v1"} and problem_key == "hh":
        if pool_key_input == "full_meta":
            raise ValueError(
                "HH continuation does not allow --adapt-pool full_meta at depth 0. "
                "Use a narrow core pool (default paop_lf_std) or run with --adapt-continuation-mode legacy."
            )
        core_key = str(pool_key_input if pool_key_input is not None else "paop_lf_std")
        core_pool, _core_method = _build_hh_pool_by_key(core_key)
        residual_pool, _residual_method = _build_hh_pool_by_key("full_meta")
        seen_sig = {_polynomial_signature(op.polynomial) for op in core_pool}
        residual_unique: list[AnsatzTerm] = []
        for op in residual_pool:
            sig = _polynomial_signature(op.polynomial)
            if sig in seen_sig:
                continue
            seen_sig.add(sig)
            residual_unique.append(op)
        pool = list(core_pool) + list(residual_unique)
        phase1_core_limit = int(len(core_pool))
        phase1_residual_indices = set(range(int(phase1_core_limit), int(len(pool))))
        pool_stage_family = (["core"] * int(phase1_core_limit)) + (["residual"] * int(len(residual_unique)))
        pool_family_ids = ([str(core_key)] * int(phase1_core_limit)) + (["full_meta"] * int(len(residual_unique)))
        method_name = f"hardcoded_adapt_vqe_{str(continuation_mode)}_hh"
        pool_key = str(continuation_mode)
    else:
        pool_key = str(pool_key_input if pool_key_input is not None else ("uccsd" if problem_key == "hubbard" else "full_meta"))
        if problem_key == "hh":
            pool, method_name = _build_hh_pool_by_key(pool_key)
        else:
            if pool_key == "uccsd":
                pool = _build_uccsd_pool(int(num_sites), num_particles, str(ordering))
                method_name = "hardcoded_adapt_vqe_uccsd"
            elif pool_key == "cse":
                pool = _build_cse_pool(
                    int(num_sites),
                    str(ordering),
                    float(t),
                    float(u),
                    float(dv),
                    str(boundary),
                )
                method_name = "hardcoded_adapt_vqe_cse"
            elif pool_key == "full_hamiltonian":
                pool = _build_full_hamiltonian_pool(h_poly)
                method_name = "hardcoded_adapt_vqe_full_hamiltonian"
            elif pool_key == "hva":
                raise ValueError(
                    "For problem='hubbard', pool='hva' is not valid. "
                    "Use uccsd, cse, or full_hamiltonian."
                )
            elif pool_key == "full_meta":
                raise ValueError("Pool 'full_meta' is only valid for problem='hh'.")
            elif pool_key == "uccsd_paop_lf_full":
                raise ValueError("Pool 'uccsd_paop_lf_full' is only valid for problem='hh'.")
            else:
                raise ValueError(f"Unsupported adapt pool '{adapt_pool}'.")
        pool_stage_family = [str(pool_key)] * int(len(pool))
        pool_family_ids = [str(pool_key)] * int(len(pool))

    if len(pool) == 0:
        raise ValueError(f"ADAPT pool '{pool_key}' produced no operators for problem='{problem_key}'.")
    _ai_log(
        "hardcoded_adapt_pool_built",
        pool_type=str(pool_key),
        pool_size=int(len(pool)),
        continuation_mode=str(continuation_mode),
        phase1_core_size=(
            int(phase1_core_limit)
            if continuation_mode in {"phase1_v1", "phase2_v1", "phase3_v1"} and problem_key == "hh"
            else None
        ),
        phase1_residual_size=(
            int(len(phase1_residual_indices))
            if continuation_mode in {"phase1_v1", "phase2_v1", "phase3_v1"} and problem_key == "hh"
            else None
        ),
    )

    phase1_enabled = bool(continuation_mode in {"phase1_v1", "phase2_v1", "phase3_v1"} and problem_key == "hh")
    phase2_enabled = bool(continuation_mode in {"phase2_v1", "phase3_v1"} and problem_key == "hh")
    phase3_enabled = bool(continuation_mode == "phase3_v1" and problem_key == "hh")
    qpb = int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding))) if problem_key == "hh" else 1
    pool_symmetry_specs: list[dict[str, Any] | None] = [None] * int(len(pool))
    pool_generator_registry: dict[str, dict[str, Any]] = {}
    phase3_split_events: list[dict[str, Any]] = []
    phase3_input_motif_library: dict[str, Any] | None = None
    phase3_runtime_split_summary: dict[str, Any] = {
        "mode": (str(phase3_runtime_split_mode_key) if phase3_enabled else "off"),
        "probed_parent_count": 0,
        "evaluated_child_count": 0,
        "selected_child_count": 0,
        "selected_child_labels": [],
    }
    phase3_motif_usage: dict[str, Any] = {
        "enabled": False,
        "source_json": (str(phase3_motif_source_json) if phase3_motif_source_json is not None else None),
        "source_tag": None,
        "seeded_labels": [],
        "seeded_generator_ids": [],
        "seeded_motif_ids": [],
        "selected_match_count": 0,
    }
    phase3_rescue_history: list[dict[str, Any]] = []
    phase3_exact_reference_state: np.ndarray | None = None
    if phase3_enabled:
        pool_symmetry_specs = [
            dict(
                build_symmetry_spec(
                    family_id=str(pool_family_ids[idx] if idx < len(pool_family_ids) else "unknown"),
                    mitigation_mode=str(phase3_symmetry_mitigation_mode_key),
                ).__dict__
            )
            for idx in range(len(pool))
        ]
        pool_generator_registry = build_pool_generator_registry(
            terms=pool,
            family_ids=pool_family_ids,
            num_sites=int(num_sites),
            ordering=str(ordering),
            qpb=int(max(1, qpb)),
            symmetry_specs=pool_symmetry_specs,
            split_policy=("deliberate_split" if bool(paop_split_paulis) else "preserve"),
        )
        if phase3_motif_source_json is not None:
            phase3_input_motif_library = load_motif_library_from_json(Path(phase3_motif_source_json))
            if phase3_input_motif_library is not None:
                phase3_motif_usage["enabled"] = True
                phase3_motif_usage["source_tag"] = str(phase3_input_motif_library.get("source_tag", "payload"))
        if bool(phase3_enable_rescue):
            phase3_exact_reference_state = _exact_reference_state_for_hh(
                num_sites=int(num_sites),
                num_particles=num_particles,
                indexing=str(ordering),
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                t=float(t),
                u=float(u),
                dv=float(dv),
                omega0=float(omega0),
                g_ep=float(g_ep),
                boundary=str(boundary),
            )
        _ai_log(
            "hardcoded_adapt_phase3_registry_ready",
            pool_size=int(len(pool)),
            generator_count=int(len(pool_generator_registry)),
            motif_source_enabled=bool(phase3_input_motif_library is not None),
            rescue_exact_state_available=bool(phase3_exact_reference_state is not None),
        )

    compile_cache_t0 = time.perf_counter()
    pauli_action_cache: dict[str, CompiledPauliAction] = {}
    h_compiled = _compile_polynomial_action(
        h_poly,
        pauli_action_cache=pauli_action_cache,
    )
    pool_compiled = [
        _compile_polynomial_action(
            op.polynomial,
            pauli_action_cache=pauli_action_cache,
        )
        for op in pool
    ]
    compile_cache_elapsed_s = float(time.perf_counter() - compile_cache_t0)
    pool_compiled_terms_total = int(sum(len(compiled_poly.terms) for compiled_poly in pool_compiled))
    _ai_log(
        "hardcoded_adapt_compiled_cache_ready",
        pool_size=int(len(pool)),
        h_terms=int(len(h_compiled.terms)),
        pool_terms_total=pool_compiled_terms_total,
        unique_pauli_actions=int(len(pauli_action_cache)),
        compile_elapsed_s=compile_cache_elapsed_s,
    )
    _ai_log(
        "hardcoded_adapt_compile_timing",
        pool_size=int(len(pool)),
        pool_terms_total=pool_compiled_terms_total,
        unique_pauli_actions=int(len(pauli_action_cache)),
        compile_elapsed_s=compile_cache_elapsed_s,
    )

    def _build_compiled_executor(ops: list[AnsatzTerm]) -> CompiledAnsatzExecutor:
        return CompiledAnsatzExecutor(
            ops,
            coefficient_tolerance=1e-12,
            ignore_identity=True,
            sort_terms=True,
            pauli_action_cache=pauli_action_cache,
        )

    # ADAPT-VQE main loop
    selected_ops: list[AnsatzTerm] = []
    theta = np.zeros(0, dtype=float)
    selected_executor: CompiledAnsatzExecutor | None = None
    history: list[dict[str, Any]] = []
    nfev_total = 0
    stop_reason = "max_depth"

    scipy_minimize = None
    if adapt_inner_optimizer_key == "COBYLA":
        from scipy.optimize import minimize as scipy_minimize

    # Pool availability tracking (for no-repeat mode)
    available_indices = (
        set(range(int(phase1_core_limit)))
        if phase1_enabled
        else set(range(len(pool)))
    )
    selection_counts = np.zeros(len(pool), dtype=np.int64)
    phase1_stage_cfg = StageControllerConfig(
        plateau_patience=int(max(1, phase1_plateau_patience)),
        weak_drop_threshold=(
            float(adapt_drop_floor)
            if bool(drop_policy_enabled)
            else float(max(float(eps_energy), 1e-12))
        ),
        probe_margin_ratio=float(max(0.0, phase1_trough_margin_ratio)),
        max_probe_positions=int(max(1, phase1_probe_max_positions)),
        append_admit_threshold=0.05,
        family_repeat_patience=int(max(1, phase1_plateau_patience)),
    )
    phase1_stage = StageController(phase1_stage_cfg)
    if phase1_enabled:
        phase1_stage.start_with_seed()
    phase1_score_cfg = SimpleScoreConfig(
        lambda_F=float(phase1_lambda_F),
        lambda_compile=float(phase1_lambda_compile),
        lambda_measure=float(phase1_lambda_measure),
        lambda_leak=float(phase1_lambda_leak),
        z_alpha=float(phase1_score_z_alpha),
    )
    phase1_compile_oracle = Phase1CompileCostOracle()
    phase1_measure_cache = MeasurementCacheAudit(nominal_shots_per_group=1)
    phase2_score_cfg = FullScoreConfig(
        z_alpha=float(phase1_score_z_alpha),
        lambda_F=float(phase1_lambda_F),
        lambda_H=float(max(1e-12, phase2_lambda_H)),
        rho=float(max(1e-6, phase2_rho)),
        gamma_N=float(max(0.0, phase2_gamma_N)),
        shortlist_fraction=float(max(0.05, phase2_shortlist_fraction)),
        shortlist_size=int(max(1, phase2_shortlist_size)),
        batch_target_size=int(max(1, phase2_batch_target_size)),
        batch_size_cap=int(max(1, phase2_batch_size_cap)),
        batch_near_degenerate_ratio=float(max(0.0, min(1.0, phase2_batch_near_degenerate_ratio))),
        lifetime_cost_mode=(
            str(phase3_lifetime_cost_mode_key)
            if phase3_enabled and str(phase3_lifetime_cost_mode_key) != "off"
            else "off"
        ),
        remaining_evaluations_proxy_mode=(
            "remaining_depth"
            if phase3_enabled and str(phase3_lifetime_cost_mode_key) != "off"
            else "none"
        ),
    )
    phase2_novelty_oracle = Phase2NoveltyOracle()
    phase2_curvature_oracle = Phase2CurvatureOracle()
    phase2_memory_adapter = Phase2OptimizerMemoryAdapter()
    phase2_compiled_term_cache: dict[str, Any] = {}
    phase2_optimizer_memory = phase2_memory_adapter.unavailable(
        method=str(adapt_inner_optimizer_key),
        parameter_count=int(theta.size),
        reason="pre_seed_state",
    )
    phase1_residual_opened = False
    phase1_last_probe_reason = "none"
    phase1_last_positions_considered: list[int] = []
    phase1_last_trough_detected = False
    phase1_last_trough_probe_triggered = False
    phase1_last_selected_score: float | None = None
    phase1_features_history: list[dict[str, Any]] = []
    phase1_stage_events: list[dict[str, Any]] = []
    phase1_scaffold_pre_prune: dict[str, Any] | None = None
    phase2_last_shortlist_records: list[dict[str, Any]] = []
    phase2_last_batch_selected = False
    phase2_last_batch_penalty_total = 0.0
    phase2_last_optimizer_memory_reused = False
    phase2_last_optimizer_memory_source = "unavailable"
    phase2_last_shortlist_eval_records: list[dict[str, Any]] = []

    energy_current, _ = energy_via_one_apply(psi_ref, h_compiled)
    energy_current = float(energy_current)
    nfev_total += 1
    _ai_log("hardcoded_adapt_initial_energy", energy=energy_current)
    if exact_gs_override is None:
        exact_gs = _exact_gs_energy_for_problem(
            h_poly,
            problem=problem_key,
            num_sites=int(num_sites),
            num_particles=num_particles,
            indexing=str(ordering),
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            t=float(t),
            u=float(u),
            dv=float(dv),
            omega0=float(omega0),
            g_ep=float(g_ep),
            boundary=str(boundary),
        )
    else:
        exact_gs = float(exact_gs_override)
        _ai_log("hardcoded_adapt_exact_override_used", exact_gs=exact_gs)
    drop_prev_delta_abs = float(abs(energy_current - exact_gs))
    drop_plateau_hits = 0
    eps_energy_low_streak = 0

    # HH preconditioning: optimize a compact boson-quadrature e-ph seed block
    # before greedy ADAPT selection. This helps avoid the weak-coupling basin
    # when g is moderate/strong.
    if (
        (not disable_hh_seed)
        and problem_key == "hh"
        and abs(float(g_ep)) > 1e-15
    ):
        n_sites = int(num_sites)
        boson_bits = n_sites * int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
        seed_indices: list[int] = []
        for idx, op in enumerate(pool):
            label = str(op.label)
            if not label.startswith("hh_termwise_ham_quadrature_term("):
                continue
            op_terms = op.polynomial.return_polynomial()
            if not op_terms:
                continue
            pw = str(op_terms[0].pw2strng())
            has_boson_y = any(ch == "y" for ch in pw[:boson_bits])
            has_electron_z = ("z" in pw[boson_bits:])
            if has_boson_y and has_electron_z:
                seed_indices.append(idx)

        if seed_indices:
            seed_ops = [pool[i] for i in seed_indices]
            theta_seed0 = np.zeros(len(seed_ops), dtype=float)
            seed_executor = (
                _build_compiled_executor(seed_ops)
                if adapt_state_backend_key == "compiled"
                else None
            )
            seed_opt_t0 = time.perf_counter()
            seed_cobyla_last_hb_t = seed_opt_t0
            seed_cobyla_nfev_so_far = 0
            seed_cobyla_best_fun = float("inf")

            def _seed_obj(x: np.ndarray) -> float:
                nonlocal seed_cobyla_last_hb_t, seed_cobyla_nfev_so_far, seed_cobyla_best_fun
                if seed_executor is not None:
                    psi_seed = seed_executor.prepare_state(np.asarray(x, dtype=float), psi_ref)
                    seed_energy, _ = energy_via_one_apply(psi_seed, h_compiled)
                    seed_energy_val = float(seed_energy)
                else:
                    seed_energy_val = _adapt_energy_fn(
                        h_poly,
                        psi_ref,
                        seed_ops,
                        x,
                        h_compiled=h_compiled,
                    )
                if adapt_inner_optimizer_key == "COBYLA":
                    seed_cobyla_nfev_so_far += 1
                    if seed_energy_val < seed_cobyla_best_fun:
                        seed_cobyla_best_fun = float(seed_energy_val)
                    now = time.perf_counter()
                    if (now - seed_cobyla_last_hb_t) >= float(adapt_spsa_progress_every_s):
                        _ai_log(
                            "hardcoded_adapt_cobyla_heartbeat",
                            stage="hh_seed_preopt",
                            depth=0,
                            nfev_opt_so_far=int(seed_cobyla_nfev_so_far),
                            best_fun=float(seed_cobyla_best_fun),
                            delta_abs_best=(
                                float(abs(seed_cobyla_best_fun - exact_gs))
                                if math.isfinite(seed_cobyla_best_fun)
                                else None
                            ),
                            elapsed_opt_s=float(now - seed_opt_t0),
                        )
                        seed_cobyla_last_hb_t = now
                return float(seed_energy_val)

            seed_maxiter = int(max(100, min(int(maxiter), 600)))
            if adapt_inner_optimizer_key == "SPSA":
                seed_last_hb_t = seed_opt_t0

                def _seed_spsa_callback(ev: dict[str, Any]) -> None:
                    nonlocal seed_last_hb_t
                    now = time.perf_counter()
                    if (now - seed_last_hb_t) < float(adapt_spsa_progress_every_s):
                        return
                    seed_best = float(ev.get("best_fun", float("nan")))
                    _ai_log(
                        "hardcoded_adapt_spsa_heartbeat",
                        stage="hh_seed_preopt",
                        depth=0,
                        iter=int(ev.get("iter", 0)),
                        nfev_opt_so_far=int(ev.get("nfev_so_far", 0)),
                        best_fun=seed_best,
                        delta_abs_best=float(abs(seed_best - exact_gs)) if math.isfinite(seed_best) else None,
                        elapsed_opt_s=float(now - seed_opt_t0),
                    )
                    seed_last_hb_t = now

                seed_result = spsa_minimize(
                    fun=_seed_obj,
                    x0=theta_seed0,
                    maxiter=int(seed_maxiter),
                    seed=int(seed) + 90000,
                    a=float(adapt_spsa_a),
                    c=float(adapt_spsa_c),
                    alpha=float(adapt_spsa_alpha),
                    gamma=float(adapt_spsa_gamma),
                    A=float(adapt_spsa_A),
                    bounds=None,
                    project="none",
                    eval_repeats=int(adapt_spsa_eval_repeats),
                    eval_agg=str(adapt_spsa_eval_agg_key),
                    avg_last=int(adapt_spsa_avg_last),
                    callback=_seed_spsa_callback,
                    callback_every=int(adapt_spsa_callback_every),
                )
                seed_theta = np.asarray(seed_result.x, dtype=float)
                seed_energy = float(seed_result.fun)
                seed_nfev = int(seed_result.nfev)
                seed_nit = int(seed_result.nit)
                seed_success = bool(seed_result.success)
                seed_message = str(seed_result.message)
            else:
                if scipy_minimize is None:
                    raise RuntimeError("SciPy minimize is unavailable for COBYLA ADAPT inner optimizer.")
                seed_result = scipy_minimize(
                    _seed_obj,
                    theta_seed0,
                    method="COBYLA",
                    options={"maxiter": int(seed_maxiter), "rhobeg": 0.3},
                )
                seed_theta = np.asarray(seed_result.x, dtype=float)
                seed_energy = float(seed_result.fun)
                seed_nfev = int(getattr(seed_result, "nfev", 0))
                seed_nit = int(getattr(seed_result, "nit", 0))
                seed_success = bool(getattr(seed_result, "success", False))
                seed_message = str(getattr(seed_result, "message", ""))
            nfev_total += int(seed_nfev)

            selected_ops = list(seed_ops)
            theta = np.asarray(seed_theta, dtype=float)
            if phase2_enabled:
                if adapt_inner_optimizer_key == "SPSA":
                    phase2_optimizer_memory = phase2_memory_adapter.from_result(
                        seed_result,
                        method=str(adapt_inner_optimizer_key),
                        parameter_count=int(len(seed_ops)),
                        source="hh_seed_preopt",
                    )
                else:
                    phase2_optimizer_memory = phase2_memory_adapter.unavailable(
                        method=str(adapt_inner_optimizer_key),
                        parameter_count=int(len(seed_ops)),
                        reason="non_spsa_seed_preopt",
                    )
            if not allow_repeats:
                for idx in seed_indices:
                    available_indices.discard(idx)
            energy_current = float(seed_energy)
            selected_executor = (
                seed_executor
                if seed_executor is not None
                else None
            )
            _ai_log(
                "hardcoded_adapt_hh_seed_preopt",
                num_seed_ops=int(len(seed_ops)),
                seed_opt_method=str(adapt_inner_optimizer_key),
                seed_energy=float(seed_energy),
                seed_nfev=int(seed_nfev),
                seed_nit=int(seed_nit),
                seed_success=bool(seed_success),
                seed_message=str(seed_message),
            )
            if phase1_enabled:
                phase1_stage.begin_core()
                phase1_stage_events.append(
                    {
                        "depth": 0,
                        "stage_name": "seed",
                        "reason": "seed_complete",
                        "num_seed_ops": int(len(seed_ops)),
                    }
                )
    if phase1_enabled and phase1_stage.stage_name == "seed":
        phase1_stage.begin_core()
        phase1_stage_events.append(
            {
                "depth": 0,
                "stage_name": "seed",
                "reason": "seed_skipped_or_empty",
                "num_seed_ops": 0,
            }
        )
    if phase2_enabled and int(theta.size) > 0 and int(phase2_optimizer_memory.get("parameter_count", 0)) != int(theta.size):
        phase2_optimizer_memory = phase2_memory_adapter.unavailable(
            method=str(adapt_inner_optimizer_key),
            parameter_count=int(theta.size),
            reason="post_seed_memory_resize",
        )

    if phase3_enabled and isinstance(phase3_input_motif_library, Mapping):
        motif_seed_records = select_tiled_generators_from_library(
            motif_library=phase3_input_motif_library,
            registry_by_label=pool_generator_registry,
            target_num_sites=int(num_sites),
            excluded_labels=[str(op.label) for op in selected_ops],
            max_seed=4,
        )
        label_to_indices: dict[str, list[int]] = {}
        for idx_pool, op_pool in enumerate(pool):
            label_to_indices.setdefault(str(op_pool.label), []).append(int(idx_pool))
        seeded_labels_now: list[str] = []
        seeded_generator_ids_now: list[str] = []
        seeded_motif_ids_now: list[str] = []
        for rec in motif_seed_records:
            label_seed = str(rec.get("candidate_label", ""))
            idx_list = label_to_indices.get(label_seed, [])
            idx_seed = None
            for idx_candidate in idx_list:
                if bool(allow_repeats) or int(idx_candidate) in available_indices:
                    idx_seed = int(idx_candidate)
                    break
            if idx_seed is None:
                continue
            if phase2_enabled:
                phase2_optimizer_memory = phase2_memory_adapter.remap_insert(
                    phase2_optimizer_memory,
                    position_id=int(theta.size),
                    count=1,
                )
            selected_ops, theta = _splice_candidate_at_position(
                ops=selected_ops,
                theta=np.asarray(theta, dtype=float),
                op=pool[int(idx_seed)],
                position_id=int(theta.size),
                init_theta=0.0,
            )
            selection_counts[int(idx_seed)] += 1
            if not allow_repeats:
                available_indices.discard(int(idx_seed))
            seeded_labels_now.append(str(label_seed))
            generator_meta = rec.get("generator_metadata", None)
            if isinstance(generator_meta, Mapping) and generator_meta.get("generator_id") is not None:
                seeded_generator_ids_now.append(str(generator_meta.get("generator_id")))
            motif_meta = rec.get("motif_metadata", None)
            if isinstance(motif_meta, Mapping):
                for motif_id in motif_meta.get("motif_ids", []):
                    seeded_motif_ids_now.append(str(motif_id))
        if seeded_labels_now:
            phase3_motif_usage["seeded_labels"] = [str(x) for x in seeded_labels_now]
            phase3_motif_usage["seeded_generator_ids"] = [str(x) for x in seeded_generator_ids_now]
            phase3_motif_usage["seeded_motif_ids"] = [str(x) for x in seeded_motif_ids_now]
            phase1_stage_events.append(
                {
                    "depth": 0,
                    "stage_name": str(phase1_stage.stage_name if phase1_enabled else "legacy"),
                    "reason": "motif_seed_injected",
                    "seeded_labels": [str(x) for x in seeded_labels_now],
                }
            )
            if adapt_state_backend_key == "compiled":
                selected_executor = _build_compiled_executor(selected_ops)
            _ai_log(
                "hardcoded_adapt_phase3_motif_seeded",
                seeded_count=int(len(seeded_labels_now)),
                seeded_labels=[str(x) for x in seeded_labels_now],
                source_tag=str(phase3_motif_usage.get("source_tag")),
            )

    rescue_cfg = RescueConfig(enabled=bool(phase3_enable_rescue))

    def _phase3_try_rescue(
        *,
        psi_current_state: np.ndarray,
        shortlist_eval_records: list[dict[str, Any]],
        selected_position_append: int,
        history_rows: list[dict[str, Any]],
        trough_detected_now: bool,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        diagnostic = {
            "enabled": bool(phase3_enable_rescue),
            "triggered": False,
            "reason": "disabled",
            "ranked": [],
            "selected_label": None,
            "selected_position": None,
            "overlap_gain": 0.0,
        }
        trigger_on, trigger_reason = should_trigger_rescue(
            enabled=bool(phase3_enable_rescue),
            exact_state_available=bool(phase3_exact_reference_state is not None),
            residual_opened=bool(phase1_residual_opened),
            trough_detected=bool(trough_detected_now),
            history=history_rows,
            shortlist_records=shortlist_eval_records,
            cfg=rescue_cfg,
        )
        diagnostic["reason"] = str(trigger_reason)
        if not bool(trigger_on):
            return None, diagnostic
        diagnostic["triggered"] = True
        psi_exact_ref = np.asarray(phase3_exact_reference_state, dtype=complex)
        overlap_current = float(abs(np.vdot(psi_exact_ref, np.asarray(psi_current_state, dtype=complex))) ** 2)
        probe_theta_by_key: dict[tuple[int, int], float] = {}

        def _overlap_gain(rec: Mapping[str, Any]) -> float:
            idx_sel = int(rec.get("candidate_pool_index", -1))
            pos_sel = int(rec.get("position_id", selected_position_append))
            if idx_sel < 0 or idx_sel >= len(pool):
                return 0.0
            best_gain_local = 0.0
            best_theta_local = 0.0
            for theta_probe in (float(finite_angle), -float(finite_angle)):
                ops_trial, theta_trial = _splice_candidate_at_position(
                    ops=selected_ops,
                    theta=np.asarray(theta, dtype=float),
                    op=pool[int(idx_sel)],
                    position_id=int(pos_sel),
                    init_theta=float(theta_probe),
                )
                if adapt_state_backend_key == "compiled":
                    psi_trial = _build_compiled_executor(ops_trial).prepare_state(theta_trial, psi_ref)
                else:
                    psi_trial = _prepare_adapt_state(psi_ref, ops_trial, theta_trial)
                overlap_trial = float(abs(np.vdot(psi_exact_ref, np.asarray(psi_trial, dtype=complex))) ** 2)
                gain = float(overlap_trial - overlap_current)
                if gain > best_gain_local:
                    best_gain_local = float(gain)
                    best_theta_local = float(theta_probe)
            probe_theta_by_key[(int(idx_sel), int(pos_sel))] = float(best_theta_local)
            return float(best_gain_local)

        best_record, rescue_meta = rank_rescue_candidates(
            records=shortlist_eval_records,
            overlap_gain_fn=_overlap_gain,
            cfg=rescue_cfg,
        )
        diagnostic.update(
            {
                "reason": str(rescue_meta.get("reason", trigger_reason)),
                "ranked": [dict(x) for x in rescue_meta.get("ranked", [])],
            }
        )
        if best_record is None:
            return None, diagnostic
        idx_best = int(best_record.get("candidate_pool_index", -1))
        pos_best = int(best_record.get("position_id", selected_position_append))
        diagnostic.update(
            {
                "selected_label": str(best_record.get("candidate_label", "")),
                "selected_position": int(pos_best),
                "overlap_gain": float(best_record.get("overlap_gain", 0.0)),
                "init_theta": float(probe_theta_by_key.get((int(idx_best), int(pos_best)), 0.0)),
            }
        )
        best_out = dict(best_record)
        best_out["rescue_init_theta"] = float(probe_theta_by_key.get((int(idx_best), int(pos_best)), 0.0))
        return best_out, diagnostic

    for depth in range(int(max_depth)):
        iter_t0 = time.perf_counter()

        # 1) Compute the current state
        if adapt_state_backend_key == "compiled":
            if len(selected_ops) == 0:
                psi_current = np.array(psi_ref, copy=True)
            else:
                if selected_executor is None:
                    selected_executor = _build_compiled_executor(selected_ops)
                psi_current = selected_executor.prepare_state(theta, psi_ref)
        else:
            psi_current = _prepare_adapt_state(psi_ref, selected_ops, theta)
        energy_current, hpsi_current = energy_via_one_apply(psi_current, h_compiled)
        energy_current = float(energy_current)

        # 2) Compute commutator gradients for all pool operators
        gradient_eval_t0 = time.perf_counter()
        gradients = np.zeros(len(pool), dtype=float)
        grad_magnitudes = np.zeros(len(pool), dtype=float)
        for i in available_indices:
            apsi = _apply_compiled_polynomial(psi_current, pool_compiled[i])
            gradients[i] = adapt_commutator_grad_from_hpsi(hpsi_current, apsi)
            grad_magnitudes[i] = abs(float(gradients[i]))
        if bool(adapt_gradient_parity_check) and available_indices:
            parity_idx = max(available_indices, key=lambda idx: grad_magnitudes[int(idx)])
            grad_old = _commutator_gradient(
                h_poly,
                pool[int(parity_idx)],
                psi_current,
                h_compiled=h_compiled,
                pool_compiled=pool_compiled[int(parity_idx)],
            )
            grad_new = float(gradients[int(parity_idx)])
            rel_err = abs(grad_new - grad_old) / max(abs(grad_new), abs(grad_old), 1e-15)
            if rel_err > float(_ADAPT_GRADIENT_PARITY_RTOL):
                raise AssertionError(
                    "ADAPT gradient parity check failed: "
                    f"depth={depth + 1}, idx={int(parity_idx)}, grad_new={grad_new:.16e}, "
                    f"grad_old={float(grad_old):.16e}, rel_err={float(rel_err):.3e}, "
                    f"rtol={_ADAPT_GRADIENT_PARITY_RTOL:.1e}"
                )
        gradient_eval_elapsed_s = float(time.perf_counter() - gradient_eval_t0)
        _ai_log(
            "hardcoded_adapt_gradient_timing",
            depth=int(depth + 1),
            available_count=int(len(available_indices)),
            gradient_eval_elapsed_s=float(gradient_eval_elapsed_s),
        )

        # 2b) Select candidate (legacy argmax or phase1_v1 simple score).
        selected_position = int(theta.size)
        stage_name = "legacy"
        phase1_feature_selected: dict[str, Any] | None = None
        phase1_stage_transition_reason = "legacy"
        append_position = int(theta.size)
        phase1_append_best_score = float("-inf")
        phase2_selected_records: list[dict[str, Any]] = []
        phase2_last_shortlist_records = []
        phase2_last_shortlist_eval_records = []
        phase2_last_batch_selected = False
        phase2_last_batch_penalty_total = 0.0
        if available_indices:
            max_grad = float(max(float(grad_magnitudes[i]) for i in available_indices))
        else:
            max_grad = 0.0
        if phase1_enabled and available_indices:
            stage_name = str(phase1_stage.stage_name)
            append_position = int(theta.size)
            available_sorted = sorted(list(available_indices), key=lambda i: -float(grad_magnitudes[i]))
            shortlist = available_sorted[: min(len(available_sorted), 64)]
            current_active_window_for_probe, _probe_window_name = _resolve_reopt_active_indices(
                policy=str(adapt_reopt_policy_key),
                n=int(max(1, append_position)),
                theta=(np.asarray(theta, dtype=float) if append_position > 0 else np.zeros(1, dtype=float)),
                window_size=int(adapt_window_size_val),
                window_topk=int(adapt_window_topk_val),
                periodic_full_refit_triggered=False,
            )

            def _evaluate_phase1_positions(
                positions_considered_local: list[int],
                *,
                trough_probe_triggered_local: bool,
            ) -> dict[str, Any]:
                best_score_local = float("-inf")
                best_idx_local = int(shortlist[0]) if shortlist else int(max(available_indices))
                best_position_local = int(append_position)
                best_feat_local: dict[str, Any] | None = None
                append_best_score_local = float("-inf")
                append_best_g_lcb_local = 0.0
                append_best_family_local = ""
                best_non_append_score_local = float("-inf")
                best_non_append_g_lcb_local = 0.0
                records_local: list[dict[str, Any]] = []
                for idx in shortlist:
                    for pos in positions_considered_local:
                        active_window_guess = _predict_reopt_window_for_position(
                            theta=np.asarray(theta, dtype=float),
                            position_id=int(pos),
                            policy=str(adapt_reopt_policy_key),
                            window_size=int(adapt_window_size_val),
                            window_topk=int(adapt_window_topk_val),
                            periodic_full_refit_triggered=False,
                        )
                        compile_est = phase1_compile_oracle.estimate(
                            candidate_term_count=int(len(pool_compiled[int(idx)].terms)),
                            position_id=int(pos),
                            append_position=int(append_position),
                            refit_active_count=int(len(active_window_guess)),
                        )
                        meas_stats = phase1_measure_cache.estimate([str(pool[int(idx)].label)])
                        is_residual_candidate = bool(int(idx) in phase1_residual_indices)
                        stage_gate_open = (
                            (stage_name == "residual")
                            or (not is_residual_candidate)
                        )
                        generator_meta = (
                            pool_generator_registry.get(str(pool[int(idx)].label))
                            if phase3_enabled
                            else None
                        )
                        symmetry_spec = (
                            pool_symmetry_specs[int(idx)]
                            if phase3_enabled and int(idx) < len(pool_symmetry_specs)
                            else None
                        )
                        feat_obj = build_candidate_features(
                            stage_name=str(stage_name),
                            candidate_label=str(pool[int(idx)].label),
                            candidate_family=str(pool_family_ids[int(idx)]),
                            candidate_pool_index=int(idx),
                            position_id=int(pos),
                            append_position=int(append_position),
                            positions_considered=[int(x) for x in positions_considered_local],
                            gradient_signed=float(gradients[int(idx)]),
                            metric_proxy=float(abs(float(gradients[int(idx)]))),
                            sigma_hat=0.0,
                            refit_window_indices=[int(i) for i in active_window_guess],
                            compile_cost=compile_est,
                            measurement_stats=meas_stats,
                            leakage_penalty=(
                                float(leakage_penalty_from_spec(symmetry_spec))
                                if phase3_enabled
                                else 0.0
                            ),
                            stage_gate_open=bool(stage_gate_open),
                            leakage_gate_open=not bool(
                                isinstance(symmetry_spec, Mapping) and symmetry_spec.get("hard_guard", False)
                            ),
                            trough_probe_triggered=bool(trough_probe_triggered_local),
                            trough_detected=False,
                            cfg=phase1_score_cfg,
                            generator_metadata=(dict(generator_meta) if isinstance(generator_meta, Mapping) else None),
                            symmetry_spec=(dict(symmetry_spec) if isinstance(symmetry_spec, Mapping) else None),
                            symmetry_mode=("shared_phase3_spec" if phase3_enabled else "none"),
                            symmetry_mitigation_mode=str(phase3_symmetry_mitigation_mode_key if phase3_enabled else "off"),
                            current_depth=int(depth),
                            max_depth=int(max_depth),
                            lifetime_cost_mode=(
                                str(phase3_lifetime_cost_mode_key)
                                if phase3_enabled
                                else "off"
                            ),
                            remaining_evaluations_proxy_mode=(
                                "remaining_depth"
                                if phase3_enabled and str(phase3_lifetime_cost_mode_key) != "off"
                                else "none"
                            ),
                        )
                        window_terms, window_labels = _window_terms_for_position(
                            selected_ops=list(selected_ops),
                            refit_window_indices=[int(i) for i in active_window_guess],
                            position_id=int(pos),
                        )
                        feat = dict(feat_obj.__dict__)
                        score_val = float(feat.get("simple_score", float("-inf")))
                        records_local.append(
                            {
                                "feature": feat_obj,
                                "simple_score": float(score_val),
                                "candidate_pool_index": int(idx),
                                "position_id": int(pos),
                                "candidate_term": pool[int(idx)],
                                "window_terms": list(window_terms),
                                "window_labels": list(window_labels),
                            }
                        )
                        if int(pos) == int(append_position) and score_val > append_best_score_local:
                            append_best_score_local = float(score_val)
                            append_best_g_lcb_local = float(feat.get("g_lcb", 0.0))
                            append_best_family_local = str(feat.get("candidate_family", ""))
                        if int(pos) != int(append_position) and score_val > best_non_append_score_local:
                            best_non_append_score_local = float(score_val)
                            best_non_append_g_lcb_local = float(feat.get("g_lcb", 0.0))
                        if score_val > best_score_local:
                            best_score_local = float(score_val)
                            best_idx_local = int(idx)
                            best_position_local = int(pos)
                            best_feat_local = dict(feat)
                return {
                    "best_score": float(best_score_local),
                    "best_idx": int(best_idx_local),
                    "best_position": int(best_position_local),
                    "best_feat": (dict(best_feat_local) if isinstance(best_feat_local, dict) else None),
                    "append_best_score": float(append_best_score_local),
                    "append_best_g_lcb": float(append_best_g_lcb_local),
                    "append_best_family": str(append_best_family_local),
                    "best_non_append_score": float(best_non_append_score_local),
                    "best_non_append_g_lcb": float(best_non_append_g_lcb_local),
                    "records": list(records_local),
                }

            append_eval = _evaluate_phase1_positions(
                [int(append_position)],
                trough_probe_triggered_local=False,
            )
            phase1_append_best_score = float(append_eval["append_best_score"])
            repeated_family_flat = _phase1_repeated_family_flat(
                history=history,
                candidate_family=str(append_eval["append_best_family"]),
                patience=int(phase1_stage_cfg.family_repeat_patience),
                weak_drop_threshold=float(phase1_stage_cfg.weak_drop_threshold),
            )
            probe_on, probe_reason = should_probe_positions(
                stage_name=str(stage_name),
                drop_plateau_hits=int(drop_plateau_hits),
                max_grad=float(max_grad),
                eps_grad=float(eps_grad),
                append_score=float(phase1_append_best_score),
                finite_angle_flat=False,
                repeated_family_flat=bool(repeated_family_flat),
                cfg=phase1_stage_cfg,
            )
            positions_considered = [int(append_position)]
            score_eval = append_eval
            if probe_on:
                positions_considered = allowed_positions(
                    n_params=int(theta.size),
                    append_position=int(append_position),
                    active_window_indices=[int(i) for i in current_active_window_for_probe],
                    max_positions=int(phase1_stage_cfg.max_probe_positions),
                )
                score_eval = _evaluate_phase1_positions(
                    [int(x) for x in positions_considered],
                    trough_probe_triggered_local=True,
                )
            trough = detect_trough(
                append_score=float(score_eval["append_best_score"]),
                best_non_append_score=float(score_eval["best_non_append_score"]),
                best_non_append_g_lcb=float(score_eval["best_non_append_g_lcb"]),
                margin_ratio=float(phase1_stage_cfg.probe_margin_ratio),
                append_admit_threshold=float(phase1_stage_cfg.append_admit_threshold),
            )
            phase1_last_probe_reason = str(probe_reason)
            phase1_last_positions_considered = [int(x) for x in positions_considered]
            phase1_last_trough_detected = bool(trough)
            phase1_last_trough_probe_triggered = bool(probe_on)
            phase1_last_selected_score = float(score_eval["best_score"])
            best_feat = score_eval["best_feat"]
            best_idx = int(score_eval["best_idx"])
            selected_position = int(score_eval["best_position"])
            selection_mode = "simple_v1_probe" if bool(probe_on) else "simple_v1"
            if phase2_enabled:
                cheap_records = shortlist_records(
                    [
                        {
                            **dict(rec),
                            "feature": rec["feature"],
                            "simple_score": float(rec.get("simple_score", float("-inf"))),
                            "candidate_pool_index": int(rec.get("candidate_pool_index", -1)),
                            "position_id": int(rec.get("position_id", append_position)),
                        }
                        for rec in score_eval.get("records", [])
                    ],
                    cfg=phase2_score_cfg,
                    score_key="simple_score",
                )
                full_records: list[dict[str, Any]] = []
                for rec in cheap_records:
                    feat_base = rec.get("feature")
                    if not isinstance(feat_base, CandidateFeatures):
                        continue
                    window_terms = list(rec.get("window_terms", []))
                    window_labels = [str(x) for x in rec.get("window_labels", [])]
                    parent_label = str(rec.get("candidate_term").label)
                    parent_generator_meta = (
                        dict(feat_base.generator_metadata)
                        if isinstance(feat_base.generator_metadata, Mapping)
                        else (
                            dict(pool_generator_registry.get(parent_label, {}))
                            if phase3_enabled and isinstance(pool_generator_registry.get(parent_label), Mapping)
                            else None
                        )
                    )
                    parent_symmetry_spec = (
                        dict(feat_base.symmetry_spec)
                        if isinstance(feat_base.symmetry_spec, Mapping)
                        else (
                            dict(pool_symmetry_specs[int(feat_base.candidate_pool_index)])
                            if phase3_enabled
                            and int(feat_base.candidate_pool_index) < len(pool_symmetry_specs)
                            and isinstance(pool_symmetry_specs[int(feat_base.candidate_pool_index)], Mapping)
                            else None
                        )
                    )

                    def _full_record_for_candidate(
                        *,
                        candidate_term: AnsatzTerm,
                        candidate_label: str,
                        generator_metadata: Mapping[str, Any] | None,
                        symmetry_spec_candidate: Mapping[str, Any] | None,
                        runtime_split_mode_value: str = "off",
                        runtime_split_parent_label_value: str | None = None,
                        runtime_split_child_index_value: int | None = None,
                        runtime_split_child_count_value: int | None = None,
                    ) -> dict[str, Any]:
                        compiled_candidate = phase2_compiled_term_cache.get(str(candidate_label))
                        if compiled_candidate is None:
                            compiled_candidate = _compile_polynomial_action(
                                candidate_term.polynomial,
                                pauli_action_cache=pauli_action_cache,
                            )
                            phase2_compiled_term_cache[str(candidate_label)] = compiled_candidate
                        grad_candidate = float(
                            adapt_commutator_grad_from_hpsi(
                                hpsi_current,
                                _apply_compiled_polynomial(
                                    np.asarray(psi_current, dtype=complex),
                                    compiled_candidate,
                                ),
                            )
                        )
                        compile_est_candidate = phase1_compile_oracle.estimate(
                            candidate_term_count=int(len(compiled_candidate.terms)),
                            position_id=int(feat_base.position_id),
                            append_position=int(feat_base.append_position),
                            refit_active_count=int(len(feat_base.refit_window_indices)),
                        )
                        measurement_stats_candidate = phase1_measure_cache.estimate([str(candidate_label)])
                        feat_candidate_base = build_candidate_features(
                            stage_name=str(feat_base.stage_name),
                            candidate_label=str(candidate_label),
                            candidate_family=str(feat_base.candidate_family),
                            candidate_pool_index=int(feat_base.candidate_pool_index),
                            position_id=int(feat_base.position_id),
                            append_position=int(feat_base.append_position),
                            positions_considered=[int(x) for x in feat_base.positions_considered],
                            gradient_signed=float(grad_candidate),
                            metric_proxy=float(abs(grad_candidate)),
                            sigma_hat=float(feat_base.sigma_hat),
                            refit_window_indices=[int(i) for i in feat_base.refit_window_indices],
                            compile_cost=compile_est_candidate,
                            measurement_stats=measurement_stats_candidate,
                            leakage_penalty=(
                                float(leakage_penalty_from_spec(symmetry_spec_candidate))
                                if isinstance(symmetry_spec_candidate, Mapping)
                                else float(feat_base.leakage_penalty)
                            ),
                            stage_gate_open=bool(feat_base.stage_gate_open),
                            leakage_gate_open=bool(feat_base.leakage_gate_open),
                            trough_probe_triggered=bool(feat_base.trough_probe_triggered),
                            trough_detected=bool(feat_base.trough_detected),
                            cfg=phase1_score_cfg,
                            generator_metadata=(
                                dict(generator_metadata) if isinstance(generator_metadata, Mapping) else None
                            ),
                            symmetry_spec=(
                                dict(symmetry_spec_candidate)
                                if isinstance(symmetry_spec_candidate, Mapping)
                                else None
                            ),
                            symmetry_mode=str(feat_base.symmetry_mode),
                            symmetry_mitigation_mode=str(feat_base.symmetry_mitigation_mode),
                            motif_metadata=(
                                dict(feat_base.motif_metadata)
                                if isinstance(feat_base.motif_metadata, Mapping)
                                else None
                            ),
                            motif_bonus=float(feat_base.motif_bonus or 0.0),
                            motif_source=str(feat_base.motif_source),
                            current_depth=int(depth),
                            max_depth=int(max_depth),
                            lifetime_cost_mode=str(feat_base.lifetime_cost_mode),
                            remaining_evaluations_proxy_mode=str(feat_base.remaining_evaluations_proxy_mode),
                        )
                        if str(runtime_split_mode_value) != "off":
                            feat_candidate_base = CandidateFeatures(
                                **{
                                    **feat_candidate_base.__dict__,
                                    "runtime_split_mode": str(runtime_split_mode_value),
                                    "runtime_split_parent_label": (
                                        str(runtime_split_parent_label_value)
                                        if runtime_split_parent_label_value is not None
                                        else None
                                    ),
                                    "runtime_split_child_index": (
                                        int(runtime_split_child_index_value)
                                        if runtime_split_child_index_value is not None
                                        else None
                                    ),
                                    "runtime_split_child_count": (
                                        int(runtime_split_child_count_value)
                                        if runtime_split_child_count_value is not None
                                        else None
                                    ),
                                }
                            )
                        active_memory = phase2_memory_adapter.select_active(
                            phase2_optimizer_memory,
                            active_indices=list(feat_candidate_base.refit_window_indices),
                            source=f"adapt.depth{int(depth + 1)}.window_subset",
                        )
                        feat_full = build_full_candidate_features(
                            base_feature=feat_candidate_base,
                            psi_state=np.asarray(psi_current, dtype=complex),
                            candidate_term=candidate_term,
                            window_terms=list(window_terms),
                            window_labels=[str(x) for x in window_labels],
                            cfg=phase2_score_cfg,
                            novelty_oracle=phase2_novelty_oracle,
                            curvature_oracle=phase2_curvature_oracle,
                            compiled_cache=phase2_compiled_term_cache,
                            pauli_action_cache=pauli_action_cache,
                            optimizer_memory=active_memory,
                            motif_library=(phase3_input_motif_library if phase3_enabled else None),
                            target_num_sites=int(num_sites),
                        )
                        return {
                            **dict(rec),
                            "feature": feat_full,
                            "simple_score": float(feat_full.simple_score or float("-inf")),
                            "full_v2_score": float(feat_full.full_v2_score or float("-inf")),
                            "candidate_pool_index": int(feat_full.candidate_pool_index),
                            "position_id": int(feat_full.position_id),
                            "candidate_term": candidate_term,
                        }

                    candidate_variants = [
                        _full_record_for_candidate(
                            candidate_term=rec["candidate_term"],
                            candidate_label=parent_label,
                            generator_metadata=parent_generator_meta,
                            symmetry_spec_candidate=parent_symmetry_spec,
                        )
                    ]
                    if (
                        phase3_enabled
                        and str(phase3_runtime_split_mode_key) == "shortlist_pauli_children_v1"
                        and isinstance(parent_generator_meta, Mapping)
                        and bool(parent_generator_meta.get("is_macro_generator", False))
                    ):
                        split_children = build_runtime_split_children(
                            parent_label=str(parent_label),
                            polynomial=rec["candidate_term"].polynomial,
                            family_id=str(feat_base.candidate_family),
                            num_sites=int(num_sites),
                            ordering=str(ordering),
                            qpb=int(max(1, qpb)),
                            split_mode=str(phase3_runtime_split_mode_key),
                            parent_generator_metadata=parent_generator_meta,
                            symmetry_spec=parent_symmetry_spec,
                        )
                        if split_children:
                            phase3_runtime_split_summary["probed_parent_count"] = int(
                                phase3_runtime_split_summary.get("probed_parent_count", 0)
                            ) + 1
                        for child in split_children:
                            child_label = str(child.get("child_label"))
                            child_poly = child.get("child_polynomial")
                            child_meta = child.get("child_generator_metadata")
                            if not isinstance(child_poly, PauliPolynomial):
                                continue
                            if not isinstance(child_meta, Mapping):
                                continue
                            pool_generator_registry[str(child_label)] = dict(child_meta)
                            phase3_runtime_split_summary["evaluated_child_count"] = int(
                                phase3_runtime_split_summary.get("evaluated_child_count", 0)
                            ) + 1
                            candidate_variants.append(
                                _full_record_for_candidate(
                                    candidate_term=AnsatzTerm(
                                        label=str(child_label),
                                        polynomial=child_poly,
                                    ),
                                    candidate_label=str(child_label),
                                    generator_metadata=dict(child_meta),
                                    symmetry_spec_candidate=(
                                        dict(child_meta.get("symmetry_spec", {}))
                                        if isinstance(child_meta.get("symmetry_spec"), Mapping)
                                        else parent_symmetry_spec
                                    ),
                                    runtime_split_mode_value=str(phase3_runtime_split_mode_key),
                                    runtime_split_parent_label_value=str(parent_label),
                                    runtime_split_child_index_value=(
                                        int(child.get("child_index"))
                                        if child.get("child_index") is not None
                                        else None
                                    ),
                                    runtime_split_child_count_value=(
                                        int(child.get("child_count"))
                                        if child.get("child_count") is not None
                                        else None
                                    ),
                                )
                            )
                    candidate_variants = sorted(candidate_variants, key=_phase2_record_sort_key)
                    if candidate_variants:
                        full_records.append(dict(candidate_variants[0]))
                full_records = sorted(full_records, key=_phase2_record_sort_key)
                phase2_last_shortlist_eval_records = [dict(rec) for rec in full_records]
                phase2_last_shortlist_records = [
                    dict(rec["feature"].__dict__)
                    for rec in full_records
                    if isinstance(rec.get("feature"), CandidateFeatures)
                ]
                if full_records:
                    if bool(phase2_enable_batching) and str(stage_name) == "core":
                        compat_oracle = CompatibilityPenaltyOracle(
                            cfg=phase2_score_cfg,
                            psi_state=np.asarray(psi_current, dtype=complex),
                            compiled_cache=phase2_compiled_term_cache,
                            pauli_action_cache=pauli_action_cache,
                        )
                        phase2_selected_records, phase2_last_batch_penalty_total = greedy_batch_select(
                            full_records,
                            compat_oracle,
                            phase2_score_cfg,
                        )
                    else:
                        phase2_selected_records = [dict(full_records[0])]
                    phase2_selected_records = sorted(phase2_selected_records, key=_phase2_record_sort_key)
                    phase2_last_batch_selected = bool(len(phase2_selected_records) > 1)
                    top_feat = phase2_selected_records[0].get("feature")
                    if isinstance(top_feat, CandidateFeatures):
                        phase1_feature_selected = dict(top_feat.__dict__)
                        phase1_feature_selected["trough_detected"] = bool(trough)
                        phase1_last_selected_score = float(
                            top_feat.full_v2_score if top_feat.full_v2_score is not None else top_feat.simple_score or float("-inf")
                        )
                        best_idx = int(top_feat.candidate_pool_index)
                        selected_position = int(top_feat.position_id)
                        split_selected = bool(str(top_feat.runtime_split_mode) != "off")
                        selection_mode = (
                            "full_v2_batch_split"
                            if phase2_last_batch_selected and split_selected
                            else (
                                "full_v2_split"
                                if split_selected
                                else ("full_v2_batch" if phase2_last_batch_selected else "full_v2")
                            )
                        )
                elif best_feat is not None:
                    best_feat["trough_detected"] = bool(trough)
                    phase1_feature_selected = dict(best_feat)
            elif best_feat is not None:
                best_feat["trough_detected"] = bool(trough)
                phase1_feature_selected = dict(best_feat)
            phase1_stage_now, phase1_stage_transition_reason = phase1_stage.resolve_stage_transition(
                drop_plateau_hits=int(drop_plateau_hits),
                trough_detected=bool(trough),
                residual_opened=bool(phase1_residual_opened),
            )
            if (
                phase1_stage_now == "residual"
                and (not phase1_residual_opened)
                and len(phase1_residual_indices) > 0
            ):
                phase1_residual_opened = True
                available_indices |= set(int(i) for i in phase1_residual_indices)
                phase1_stage_events.append(
                    {
                        "depth": int(depth + 1),
                        "stage_name": "residual",
                        "reason": str(phase1_stage_transition_reason),
                    }
                )
                _ai_log(
                    "hardcoded_adapt_phase1_residual_opened",
                    depth=int(depth + 1),
                    residual_count=int(len(phase1_residual_indices)),
                )
        else:
            if allow_repeats:
                repeat_bias = 1.5
                scores = grad_magnitudes / (1.0 + repeat_bias * selection_counts.astype(float))
                best_idx = int(np.argmax(scores))
            else:
                best_idx = int(np.argmax(grad_magnitudes))
            selection_mode = "gradient"

        _ai_log(
            "hardcoded_adapt_iter",
            depth=int(depth + 1),
            max_grad=float(max_grad),
            best_op=(
                str(phase2_selected_records[0].get("candidate_term").label)
                if phase1_enabled
                and phase2_enabled
                and len(phase2_selected_records) > 0
                and phase2_selected_records[0].get("candidate_term") is not None
                else (
                    str(phase1_feature_selected.get("candidate_label"))
                    if isinstance(phase1_feature_selected, dict)
                    and phase1_feature_selected.get("candidate_label") is not None
                    else str(pool[best_idx].label)
                )
            ),
            selected_position=int(selected_position),
            stage_name=str(stage_name),
            selection_score=(float(phase1_last_selected_score) if phase1_enabled else None),
            energy=float(energy_current),
        )

        # 3) Check gradient convergence (with optional finite-angle fallback)
        if not phase1_enabled:
            selection_mode = "gradient"
        init_theta = 0.0
        fallback_scan_size = 0
        fallback_best_probe_delta_e = None
        fallback_best_probe_theta = None
        if max_grad < float(eps_grad):
            if bool(finite_angle_fallback) and available_indices:
                fallback_scan_size = int(len(available_indices))
                best_probe_energy = float(energy_current)
                best_probe_idx = None
                best_probe_theta = None
                fallback_executor_cache: dict[int, CompiledAnsatzExecutor] = {}

                for idx in available_indices:
                    trial_ops = selected_ops + [pool[idx]]
                    for trial_theta in (float(finite_angle), -float(finite_angle)):
                        trial_theta_vec = np.append(theta, trial_theta)
                        if adapt_state_backend_key == "compiled":
                            trial_executor = fallback_executor_cache.get(int(idx))
                            if trial_executor is None:
                                trial_executor = _build_compiled_executor(trial_ops)
                                fallback_executor_cache[int(idx)] = trial_executor
                            psi_trial = trial_executor.prepare_state(trial_theta_vec, psi_ref)
                            probe_energy, _ = energy_via_one_apply(psi_trial, h_compiled)
                            probe_energy = float(probe_energy)
                        else:
                            probe_energy = _adapt_energy_fn(
                                h_poly,
                                psi_ref,
                                trial_ops,
                                trial_theta_vec,
                                h_compiled=h_compiled,
                            )
                        nfev_total += 1
                        if probe_energy < best_probe_energy:
                            best_probe_energy = float(probe_energy)
                            best_probe_idx = int(idx)
                            best_probe_theta = float(trial_theta)

                fallback_best_probe_delta_e = float(best_probe_energy - energy_current)
                fallback_best_probe_theta = float(best_probe_theta) if best_probe_theta is not None else None
                _ai_log(
                    "hardcoded_adapt_fallback_scan",
                    depth=int(depth + 1),
                    scan_size=int(fallback_scan_size),
                    finite_angle=float(finite_angle),
                    best_probe_idx=(int(best_probe_idx) if best_probe_idx is not None else None),
                    best_probe_op=(str(pool[best_probe_idx].label) if best_probe_idx is not None else None),
                    best_probe_delta_e=float(fallback_best_probe_delta_e),
                )

                if (
                    best_probe_idx is not None
                    and (energy_current - best_probe_energy) > float(finite_angle_min_improvement)
                ):
                    best_idx = int(best_probe_idx)
                    selected_position = int(append_position)
                    phase1_feature_selected = None
                    phase2_selected_records = []
                    phase1_last_selected_score = None
                    phase1_last_positions_considered = [int(append_position)]
                    selection_mode = "finite_angle_fallback"
                    init_theta = float(best_probe_theta)
                    _ai_log(
                        "hardcoded_adapt_fallback_selected",
                        depth=int(depth + 1),
                        selected_idx=int(best_idx),
                        selected_op=str(pool[best_idx].label),
                        init_theta=float(init_theta),
                        probe_delta_e=float(fallback_best_probe_delta_e),
                    )
                else:
                    eps_grad_probe_allowed = bool(phase1_enabled and str(stage_name) != "residual")
                    eps_grad_trough = bool(phase1_last_trough_detected and int(selected_position) != int(append_position))
                    if eps_grad_probe_allowed and (not eps_grad_trough):
                        probe_positions_eps = allowed_positions(
                            n_params=int(theta.size),
                            append_position=int(append_position),
                            active_window_indices=[int(i) for i in current_active_window_for_probe],
                            max_positions=int(phase1_stage_cfg.max_probe_positions),
                        )
                        probe_eval_eps = _evaluate_phase1_positions(
                            [int(x) for x in probe_positions_eps],
                            trough_probe_triggered_local=True,
                        )
                        eps_grad_trough = detect_trough(
                            append_score=float(probe_eval_eps["append_best_score"]),
                            best_non_append_score=float(probe_eval_eps["best_non_append_score"]),
                            best_non_append_g_lcb=float(probe_eval_eps["best_non_append_g_lcb"]),
                            margin_ratio=float(phase1_stage_cfg.probe_margin_ratio),
                            append_admit_threshold=float(phase1_stage_cfg.append_admit_threshold),
                        )
                        if bool(eps_grad_trough) and int(probe_eval_eps["best_position"]) != int(append_position):
                            phase1_last_probe_reason = "eps_grad_flat"
                            phase1_last_positions_considered = [int(x) for x in probe_positions_eps]
                            phase1_last_trough_detected = True
                            phase1_last_trough_probe_triggered = True
                            phase1_last_selected_score = float(probe_eval_eps["best_score"])
                            phase1_feature_selected = dict(probe_eval_eps["best_feat"] or {})
                            phase2_selected_records = []
                            if phase1_feature_selected:
                                phase1_feature_selected["trough_detected"] = True
                            best_idx = int(probe_eval_eps["best_idx"])
                            selected_position = int(probe_eval_eps["best_position"])
                            selection_mode = "simple_v1_probe"
                    if not bool(eps_grad_trough):
                        rescue_record = None
                        rescue_diag: dict[str, Any] | None = None
                        if phase3_enabled:
                            rescue_record, rescue_diag = _phase3_try_rescue(
                                psi_current_state=np.asarray(psi_current, dtype=complex),
                                shortlist_eval_records=list(phase2_last_shortlist_eval_records),
                                selected_position_append=int(append_position),
                                history_rows=list(history),
                                trough_detected_now=bool(phase1_last_trough_detected),
                            )
                            if rescue_diag is not None:
                                phase3_rescue_history.append(dict(rescue_diag))
                        if isinstance(rescue_record, Mapping):
                            best_idx = int(rescue_record.get("candidate_pool_index", best_idx))
                            selected_position = int(rescue_record.get("position_id", append_position))
                            init_theta = float(rescue_record.get("rescue_init_theta", finite_angle))
                            selection_mode = "rescue_overlap"
                            feat_rescue = rescue_record.get("feature")
                            phase2_selected_records = [dict(rescue_record)] if phase2_enabled else []
                            if isinstance(feat_rescue, CandidateFeatures):
                                phase1_feature_selected = dict(feat_rescue.__dict__)
                                phase1_feature_selected["actual_fallback_mode"] = "rescue_overlap"
                                phase1_last_selected_score = float(
                                    feat_rescue.full_v2_score
                                    if feat_rescue.full_v2_score is not None
                                    else feat_rescue.simple_score or float("-inf")
                                )
                                if str(feat_rescue.runtime_split_mode) != "off":
                                    selection_mode = "rescue_overlap_split"
                            _ai_log(
                                "hardcoded_adapt_phase3_rescue_selected",
                                depth=int(depth + 1),
                                selected_idx=int(best_idx),
                                selected_op=(
                                    str(rescue_record.get("candidate_term").label)
                                    if rescue_record.get("candidate_term") is not None
                                    else str(pool[int(best_idx)].label)
                                ),
                                selected_position=int(selected_position),
                                init_theta=float(init_theta),
                                overlap_gain=float(rescue_record.get("overlap_gain", 0.0)),
                            )
                        else:
                            if bool(eps_grad_termination_enabled):
                                stop_reason = "eps_grad"
                                _ai_log(
                                    "hardcoded_adapt_converged_grad",
                                    max_grad=float(max_grad),
                                    eps_grad=float(eps_grad),
                                    fallback_attempted=True,
                                    fallback_best_probe_delta_e=float(fallback_best_probe_delta_e),
                                    finite_angle_min_improvement=float(finite_angle_min_improvement),
                                )
                                break
                            selection_mode = "eps_grad_suppressed_continue"
                            _ai_log(
                                "hardcoded_adapt_eps_grad_termination_suppressed",
                                depth=int(depth + 1),
                                max_grad=float(max_grad),
                                eps_grad=float(eps_grad),
                                fallback_attempted=True,
                                fallback_best_probe_delta_e=float(fallback_best_probe_delta_e),
                                finite_angle_min_improvement=float(finite_angle_min_improvement),
                                continuation_mode=str(continuation_mode),
                                problem=str(problem_key),
                            )
            else:
                eps_grad_trough = bool(phase1_enabled and phase1_last_trough_detected and int(selected_position) != int(append_position))
                if not bool(eps_grad_trough):
                    rescue_record = None
                    rescue_diag: dict[str, Any] | None = None
                    if phase3_enabled:
                        rescue_record, rescue_diag = _phase3_try_rescue(
                            psi_current_state=np.asarray(psi_current, dtype=complex),
                            shortlist_eval_records=list(phase2_last_shortlist_eval_records),
                            selected_position_append=int(append_position),
                            history_rows=list(history),
                            trough_detected_now=bool(phase1_last_trough_detected),
                        )
                        if rescue_diag is not None:
                            phase3_rescue_history.append(dict(rescue_diag))
                    if isinstance(rescue_record, Mapping):
                        best_idx = int(rescue_record.get("candidate_pool_index", best_idx))
                        selected_position = int(rescue_record.get("position_id", append_position))
                        init_theta = float(rescue_record.get("rescue_init_theta", finite_angle))
                        selection_mode = "rescue_overlap"
                        feat_rescue = rescue_record.get("feature")
                        phase2_selected_records = [dict(rescue_record)] if phase2_enabled else []
                        if isinstance(feat_rescue, CandidateFeatures):
                            phase1_feature_selected = dict(feat_rescue.__dict__)
                            phase1_feature_selected["actual_fallback_mode"] = "rescue_overlap"
                            phase1_last_selected_score = float(
                                feat_rescue.full_v2_score
                                if feat_rescue.full_v2_score is not None
                                else feat_rescue.simple_score or float("-inf")
                            )
                            if str(feat_rescue.runtime_split_mode) != "off":
                                selection_mode = "rescue_overlap_split"
                        _ai_log(
                            "hardcoded_adapt_phase3_rescue_selected",
                            depth=int(depth + 1),
                            selected_idx=int(best_idx),
                            selected_op=(
                                str(rescue_record.get("candidate_term").label)
                                if rescue_record.get("candidate_term") is not None
                                else str(pool[int(best_idx)].label)
                            ),
                            selected_position=int(selected_position),
                            init_theta=float(init_theta),
                            overlap_gain=float(rescue_record.get("overlap_gain", 0.0)),
                        )
                    else:
                        if bool(eps_grad_termination_enabled):
                            stop_reason = "eps_grad"
                            _ai_log("hardcoded_adapt_converged_grad", max_grad=float(max_grad), eps_grad=float(eps_grad))
                            break
                        selection_mode = "eps_grad_suppressed_continue"
                        _ai_log(
                            "hardcoded_adapt_eps_grad_termination_suppressed",
                            depth=int(depth + 1),
                            max_grad=float(max_grad),
                            eps_grad=float(eps_grad),
                            fallback_attempted=False,
                            continuation_mode=str(continuation_mode),
                            problem=str(problem_key),
                        )

        # 4) Admit selected operator (append or insertion in continuation modes).
        selected_batch_records_for_history: list[dict[str, Any]] = []
        selected_batch_labels: list[str] = []
        selected_batch_positions: list[int] = []
        selected_batch_indices: list[int] = []
        if phase1_enabled and phase2_enabled and len(phase2_selected_records) > 0:
            original_positions_seen: list[int] = []
            for rec in phase2_selected_records:
                feat_rec = rec.get("feature")
                if not isinstance(feat_rec, CandidateFeatures):
                    continue
                idx_sel = int(feat_rec.candidate_pool_index)
                pos_orig = int(feat_rec.position_id)
                pos_eff = int(pos_orig + sum(1 for prev in original_positions_seen if prev <= pos_orig))
                admitted_term = rec.get("candidate_term")
                if not isinstance(admitted_term, AnsatzTerm):
                    admitted_term = pool[int(idx_sel)]
                if phase2_enabled:
                    phase2_optimizer_memory = phase2_memory_adapter.remap_insert(
                        phase2_optimizer_memory,
                        position_id=int(pos_eff),
                        count=1,
                    )
                selected_ops, theta = _splice_candidate_at_position(
                    ops=selected_ops,
                    theta=np.asarray(theta, dtype=float),
                    op=admitted_term,
                    position_id=int(pos_eff),
                    init_theta=0.0,
                )
                if (
                    phase3_enabled
                    and str(feat_rec.runtime_split_mode) != "off"
                    and feat_rec.parent_generator_id is not None
                    and feat_rec.generator_id is not None
                ):
                    phase3_split_events.append(
                        build_split_event(
                            parent_generator_id=str(feat_rec.parent_generator_id),
                            child_generator_ids=[str(feat_rec.generator_id)],
                            reason=f"depth{int(depth + 1)}_selected",
                            split_mode=str(feat_rec.runtime_split_mode),
                        )
                    )
                    phase3_runtime_split_summary["selected_child_count"] = int(
                        phase3_runtime_split_summary.get("selected_child_count", 0)
                    ) + 1
                    phase3_runtime_split_summary["selected_child_labels"] = [
                        *list(phase3_runtime_split_summary.get("selected_child_labels", [])),
                        str(admitted_term.label),
                    ]
                original_positions_seen.append(int(pos_orig))
                selection_counts[idx_sel] += 1
                if not allow_repeats:
                    available_indices.discard(idx_sel)
                selected_batch_records_for_history.append(dict(feat_rec.__dict__))
                selected_batch_labels.append(str(admitted_term.label))
                selected_batch_positions.append(int(pos_orig))
                selected_batch_indices.append(int(idx_sel))
            if selected_batch_indices:
                best_idx = int(selected_batch_indices[0])
                selected_position = int(selected_batch_positions[0])
        elif phase1_enabled:
            if phase2_enabled:
                phase2_optimizer_memory = phase2_memory_adapter.remap_insert(
                    phase2_optimizer_memory,
                    position_id=int(selected_position),
                    count=1,
                )
            selected_ops, theta = _splice_candidate_at_position(
                ops=selected_ops,
                theta=np.asarray(theta, dtype=float),
                op=pool[int(best_idx)],
                position_id=int(selected_position),
                init_theta=float(init_theta),
            )
            selection_counts[best_idx] += 1
            if not allow_repeats:
                available_indices.discard(best_idx)
            if isinstance(phase1_feature_selected, dict):
                selected_batch_records_for_history.append(dict(phase1_feature_selected))
            selected_batch_labels.append(str(pool[int(best_idx)].label))
            selected_batch_positions.append(int(selected_position))
            selected_batch_indices.append(int(best_idx))
        else:
            selected_ops.append(pool[best_idx])
            theta = np.append(theta, float(init_theta))
            selection_counts[best_idx] += 1
            if not allow_repeats:
                available_indices.discard(best_idx)
            selected_batch_labels.append(str(pool[int(best_idx)].label))
            selected_batch_positions.append(int(selected_position))
            selected_batch_indices.append(int(best_idx))
        if adapt_state_backend_key == "compiled":
            selected_executor = _build_compiled_executor(selected_ops)
        else:
            selected_executor = None

        # 5) Re-optimize parameters with selected inner optimizer
        # Policy: 'full' re-optimizes all parameters (legacy behavior),
        #         'append_only' freezes the prefix and optimizes only the
        #         newly appended parameter.
        energy_prev = energy_current
        theta_before_opt = np.array(theta, copy=True)
        optimizer_t0 = time.perf_counter()
        cobyla_last_hb_t = optimizer_t0
        cobyla_nfev_so_far = 0
        cobyla_best_fun = float("inf")

        def _obj(x: np.ndarray) -> float:
            nonlocal cobyla_last_hb_t, cobyla_nfev_so_far, cobyla_best_fun
            if adapt_state_backend_key == "compiled":
                assert selected_executor is not None
                psi_obj = selected_executor.prepare_state(np.asarray(x, dtype=float), psi_ref)
                energy_obj, _ = energy_via_one_apply(psi_obj, h_compiled)
                energy_obj_val = float(energy_obj)
            else:
                energy_obj_val = _adapt_energy_fn(
                    h_poly,
                    psi_ref,
                    selected_ops,
                    x,
                    h_compiled=h_compiled,
                )
            if adapt_inner_optimizer_key == "COBYLA":
                cobyla_nfev_so_far += 1
                if energy_obj_val < cobyla_best_fun:
                    cobyla_best_fun = float(energy_obj_val)
                now = time.perf_counter()
                if (now - cobyla_last_hb_t) >= float(adapt_spsa_progress_every_s):
                    _ai_log(
                        "hardcoded_adapt_cobyla_heartbeat",
                        stage="depth_opt",
                        depth=int(depth + 1),
                        nfev_opt_so_far=int(cobyla_nfev_so_far),
                        best_fun=float(cobyla_best_fun),
                        delta_abs_best=(
                            float(abs(cobyla_best_fun - exact_gs))
                            if math.isfinite(cobyla_best_fun)
                            else None
                        ),
                        elapsed_opt_s=float(now - optimizer_t0),
                    )
                    cobyla_last_hb_t = now
            return float(energy_obj_val)

        # -- Resolve active reopt indices for this depth --
        n_theta = int(theta.size)
        depth_local = int(depth + 1)
        depth_cumulative = int(adapt_ref_base_depth) + int(depth_local)
        periodic_full_refit_triggered = bool(
            adapt_reopt_policy_key == "windowed"
            and adapt_full_refit_every_val > 0
            and depth_cumulative % adapt_full_refit_every_val == 0
        )
        reopt_active_indices, reopt_policy_effective = _resolve_reopt_active_indices(
            policy=adapt_reopt_policy_key,
            n=n_theta,
            theta=theta,
            window_size=adapt_window_size_val,
            window_topk=adapt_window_topk_val,
            periodic_full_refit_triggered=periodic_full_refit_triggered,
        )
        if phase1_enabled and isinstance(phase1_feature_selected, dict):
            phase1_feature_selected["refit_window_indices"] = [int(i) for i in reopt_active_indices]
        if phase1_enabled and selected_batch_records_for_history:
            for rec in selected_batch_records_for_history:
                rec["refit_window_indices"] = [int(i) for i in reopt_active_indices]
        _obj_opt, opt_x0 = _make_reduced_objective(theta, reopt_active_indices, _obj)
        phase2_active_memory = None
        phase2_last_optimizer_memory_reused = False
        phase2_last_optimizer_memory_source = "unavailable"
        if phase2_enabled and adapt_inner_optimizer_key == "SPSA":
            phase2_active_memory = phase2_memory_adapter.select_active(
                phase2_optimizer_memory,
                active_indices=list(reopt_active_indices),
                source=f"adapt.depth{int(depth + 1)}.opt_active",
            )
            phase2_last_optimizer_memory_reused = bool(phase2_active_memory.get("reused", False))
            phase2_last_optimizer_memory_source = str(phase2_active_memory.get("source", "unavailable"))

        if adapt_inner_optimizer_key == "SPSA":
            spsa_last_hb_t = optimizer_t0

            def _depth_spsa_callback(ev: dict[str, Any]) -> None:
                nonlocal spsa_last_hb_t
                now = time.perf_counter()
                if (now - spsa_last_hb_t) < float(adapt_spsa_progress_every_s):
                    return
                best_fun = float(ev.get("best_fun", float("nan")))
                _ai_log(
                    "hardcoded_adapt_spsa_heartbeat",
                    stage="depth_opt",
                    depth=int(depth + 1),
                    iter=int(ev.get("iter", 0)),
                    nfev_opt_so_far=int(ev.get("nfev_so_far", 0)),
                    best_fun=best_fun,
                    delta_abs_best=float(abs(best_fun - exact_gs)) if math.isfinite(best_fun) else None,
                    elapsed_opt_s=float(now - optimizer_t0),
                )
                spsa_last_hb_t = now

            result = spsa_minimize(
                fun=_obj_opt,
                x0=opt_x0,
                maxiter=int(maxiter),
                seed=int(seed) + int(depth),
                a=float(adapt_spsa_a),
                c=float(adapt_spsa_c),
                alpha=float(adapt_spsa_alpha),
                gamma=float(adapt_spsa_gamma),
                A=float(adapt_spsa_A),
                bounds=None,
                project="none",
                eval_repeats=int(adapt_spsa_eval_repeats),
                eval_agg=str(adapt_spsa_eval_agg_key),
                avg_last=int(adapt_spsa_avg_last),
                callback=_depth_spsa_callback,
                callback_every=int(adapt_spsa_callback_every),
                memory=(dict(phase2_active_memory) if isinstance(phase2_active_memory, Mapping) else None),
                refresh_every=0,
                precondition_mode=("diag_rms_grad" if phase2_enabled else "none"),
            )
            # Reconstruct full theta from reduced optimizer result
            if len(reopt_active_indices) == n_theta:
                theta = np.asarray(result.x, dtype=float)
            else:
                result_x = np.asarray(result.x, dtype=float).ravel()
                for k, idx in enumerate(reopt_active_indices):
                    theta[idx] = float(result_x[k])
            energy_current = float(result.fun)
            nfev_opt = int(result.nfev)
            nit_opt = int(result.nit)
            opt_success = bool(result.success)
            opt_message = str(result.message)
            if phase2_enabled:
                phase2_optimizer_memory = phase2_memory_adapter.merge_active(
                    phase2_optimizer_memory,
                    active_indices=list(reopt_active_indices),
                    active_state=phase2_memory_adapter.from_result(
                        result,
                        method=str(adapt_inner_optimizer_key),
                        parameter_count=int(len(reopt_active_indices)),
                        source=f"adapt.depth{int(depth + 1)}.spsa_result",
                    ),
                    source=f"adapt.depth{int(depth + 1)}.merge",
                )
        else:
            if scipy_minimize is None:
                raise RuntimeError("SciPy minimize is unavailable for COBYLA ADAPT inner optimizer.")
            result = scipy_minimize(
                _obj_opt,
                opt_x0,
                method="COBYLA",
                options={"maxiter": int(maxiter), "rhobeg": 0.3},
            )
            # Reconstruct full theta from reduced optimizer result
            if len(reopt_active_indices) == n_theta:
                theta = np.asarray(result.x, dtype=float)
            else:
                result_x = np.asarray(result.x, dtype=float).ravel()
                for k, idx in enumerate(reopt_active_indices):
                    theta[idx] = float(result_x[k])
            energy_current = float(result.fun)
            nfev_opt = int(getattr(result, "nfev", 0))
            nit_opt = int(getattr(result, "nit", 0))
            opt_success = bool(getattr(result, "success", False))
            opt_message = str(getattr(result, "message", ""))
            if phase2_enabled:
                phase2_optimizer_memory = phase2_memory_adapter.unavailable(
                    method=str(adapt_inner_optimizer_key),
                    parameter_count=int(theta.size),
                    reason="non_spsa_depth_opt",
                )

        # -- Depth-level non-improvement rollback guard --
        # If the optimizer returned an energy worse than the entry energy for
        # this depth, roll back to the pre-optimization parameters.  This
        # prevents stochastic optimizer noise from accumulating regressions.
        depth_rollback = False
        if float(energy_current) > float(energy_prev):
            _ai_log(
                "hardcoded_adapt_depth_rollback",
                depth=int(depth + 1),
                energy_before_opt=float(energy_prev),
                energy_after_opt=float(energy_current),
                regression=float(energy_current - energy_prev),
                opt_method=str(adapt_inner_optimizer_key),
            )
            theta = np.array(theta_before_opt, copy=True)
            energy_current = float(energy_prev)
            depth_rollback = True

        optimizer_elapsed_s = float(time.perf_counter() - optimizer_t0)
        nfev_total += int(nfev_opt)
        depth_local = int(depth + 1)
        depth_cumulative = int(adapt_ref_base_depth) + int(depth_local)
        delta_abs_prev = float(drop_prev_delta_abs)
        delta_abs_current = float(abs(energy_current - exact_gs))
        delta_abs_drop = float(delta_abs_prev - delta_abs_current)
        drop_prev_delta_abs = float(delta_abs_current)
        eps_energy_step_abs = float(abs(energy_current - energy_prev))
        eps_energy_low_step = bool(eps_energy_step_abs < float(eps_energy))
        eps_energy_gate_open = bool(depth_local >= int(eps_energy_min_extra_depth_effective))
        if eps_energy_gate_open:
            if eps_energy_low_step:
                eps_energy_low_streak += 1
            else:
                eps_energy_low_streak = 0
        else:
            eps_energy_low_streak = 0
        eps_energy_termination_condition = bool(eps_energy_gate_open) and (
            int(eps_energy_low_streak) >= int(eps_energy_patience_effective)
        )
        drop_low_signal = None
        drop_low_grad = None
        if drop_policy_enabled and int(depth_local) >= int(adapt_drop_min_depth):
            drop_low_signal = bool(delta_abs_drop < float(adapt_drop_floor))
            if float(adapt_grad_floor) >= 0.0:
                drop_low_grad = bool(float(max_grad) < float(adapt_grad_floor))
            else:
                drop_low_grad = True
            if bool(drop_low_signal) and bool(drop_low_grad):
                drop_plateau_hits += 1
            else:
                drop_plateau_hits = 0
        _ai_log(
            "hardcoded_adapt_optimizer_timing",
            depth=int(depth + 1),
            opt_method=str(adapt_inner_optimizer_key),
            nfev_opt=int(nfev_opt),
            nit_opt=int(nit_opt),
            opt_success=bool(opt_success),
            opt_message=str(opt_message),
            optimizer_elapsed_s=float(optimizer_elapsed_s),
        )

        selected_primary_label = (
            str(selected_batch_labels[0])
            if selected_batch_labels
            else (
                str(phase1_feature_selected.get("candidate_label"))
                if isinstance(phase1_feature_selected, dict)
                and phase1_feature_selected.get("candidate_label") is not None
                else str(pool[best_idx].label)
            )
        )
        selected_grad_signed_value = (
            float(phase1_feature_selected.get("g_signed"))
            if isinstance(phase1_feature_selected, dict)
            and phase1_feature_selected.get("g_signed") is not None
            else float(gradients[best_idx])
        )
        selected_grad_abs_value = (
            float(phase1_feature_selected.get("g_abs"))
            if isinstance(phase1_feature_selected, dict)
            and phase1_feature_selected.get("g_abs") is not None
            else float(grad_magnitudes[best_idx])
        )
        history_row = {
            "depth": int(depth + 1),
            "selected_op": str(selected_primary_label),
            "pool_index": int(best_idx),
            "selected_ops": [str(x) for x in selected_batch_labels],
            "selected_pool_indices": [int(x) for x in selected_batch_indices],
            "selection_mode": str(selection_mode),
            "init_theta": float(init_theta),
            "max_grad": float(max_grad),
            "selected_grad_signed": float(selected_grad_signed_value),
            "selected_grad_abs": float(selected_grad_abs_value),
            "fallback_scan_size": int(fallback_scan_size),
            "fallback_best_probe_delta_e": (
                float(fallback_best_probe_delta_e) if fallback_best_probe_delta_e is not None else None
            ),
            "fallback_best_probe_theta": (
                float(fallback_best_probe_theta) if fallback_best_probe_theta is not None else None
            ),
            "energy_before_opt": float(energy_prev),
            "energy_after_opt": float(energy_current),
            "delta_energy": float(energy_current - energy_prev),
            "delta_abs_prev": float(delta_abs_prev),
            "delta_abs_current": float(delta_abs_current),
            "delta_abs_drop_from_prev": float(delta_abs_drop),
            "opt_method": str(adapt_inner_optimizer_key),
            "reopt_policy": str(adapt_reopt_policy_key),
            "nfev_opt": int(nfev_opt),
            "nit_opt": int(nit_opt),
            "opt_success": bool(opt_success),
            "opt_message": str(opt_message),
            "gradient_eval_elapsed_s": float(gradient_eval_elapsed_s),
            "optimizer_elapsed_s": float(optimizer_elapsed_s),
            "iter_elapsed_s": float(time.perf_counter() - iter_t0),
            "drop_policy_enabled": bool(drop_policy_enabled),
            "drop_policy_source": str(stop_policy.drop_policy_source),
            "adapt_drop_floor_resolved": float(adapt_drop_floor),
            "adapt_drop_patience_resolved": int(adapt_drop_patience),
            "adapt_drop_min_depth_resolved": int(adapt_drop_min_depth),
            "adapt_grad_floor_resolved": float(adapt_grad_floor),
            "adapt_drop_floor_source": str(stop_policy.adapt_drop_floor_source),
            "adapt_drop_patience_source": str(stop_policy.adapt_drop_patience_source),
            "adapt_drop_min_depth_source": str(stop_policy.adapt_drop_min_depth_source),
            "adapt_grad_floor_source": str(stop_policy.adapt_grad_floor_source),
            "drop_low_signal": drop_low_signal,
            "drop_low_grad": drop_low_grad,
            "drop_plateau_hits": int(drop_plateau_hits),
            "depth_rollback": bool(depth_rollback),
            "depth_cumulative": int(depth_cumulative),
            "adapt_ref_base_depth": int(adapt_ref_base_depth),
            "eps_energy_step_abs": float(eps_energy_step_abs),
            "eps_energy_low_step": bool(eps_energy_low_step),
            "eps_energy_low_streak": int(eps_energy_low_streak),
            "eps_energy_gate_open": bool(eps_energy_gate_open),
            "eps_energy_min_extra_depth_effective": int(eps_energy_min_extra_depth_effective),
            "eps_energy_patience_effective": int(eps_energy_patience_effective),
            "eps_energy_termination_enabled": bool(eps_energy_termination_enabled),
            "eps_energy_termination_condition": bool(eps_energy_termination_condition),
            "eps_grad_termination_enabled": bool(eps_grad_termination_enabled),
            "eps_grad_threshold_hit": bool(max_grad < float(eps_grad)),
            "reopt_policy_effective": str(reopt_policy_effective),
            "reopt_active_indices": [int(i) for i in reopt_active_indices],
            "reopt_active_count": int(len(reopt_active_indices)),
            "reopt_periodic_full_refit_triggered": bool(periodic_full_refit_triggered),
        }
        if phase1_enabled:
            history_row.update(
                {
                    "continuation_mode": str(continuation_mode),
                    "candidate_family": str(
                        pool_family_ids[int(best_idx)] if int(best_idx) < len(pool_family_ids) else "legacy"
                    ),
                    "stage_name": str(stage_name),
                    "stage_transition_reason": str(phase1_stage_transition_reason),
                    "selected_position": int(selected_position),
                    "selected_positions": [int(x) for x in selected_batch_positions],
                    "batch_selected": bool(phase2_enabled and phase2_last_batch_selected),
                    "batch_size": int(len(selected_batch_labels)),
                    "positions_considered": [int(x) for x in phase1_last_positions_considered],
                    "score_version": (
                        str(phase1_feature_selected.get("score_version"))
                        if isinstance(phase1_feature_selected, dict)
                        else None
                    ),
                    "simple_score": (
                        float(phase1_feature_selected.get("simple_score"))
                        if isinstance(phase1_feature_selected, dict)
                        and phase1_feature_selected.get("simple_score") is not None
                        else None
                    ),
                    "metric_proxy": (
                        float(phase1_feature_selected.get("metric_proxy"))
                        if isinstance(phase1_feature_selected, dict)
                        else None
                    ),
                    "curvature_mode": (
                        str(phase1_feature_selected.get("curvature_mode"))
                        if isinstance(phase1_feature_selected, dict)
                        else None
                    ),
                    "novelty_mode": (
                        str(phase1_feature_selected.get("novelty_mode"))
                        if isinstance(phase1_feature_selected, dict)
                        else None
                    ),
                    "novelty": (
                        phase1_feature_selected.get("novelty")
                        if isinstance(phase1_feature_selected, dict)
                        else None
                    ),
                    "g_lcb": (
                        float(phase1_feature_selected.get("g_lcb"))
                        if isinstance(phase1_feature_selected, dict)
                        else None
                    ),
                    "refit_window_indices": (
                        [int(i) for i in phase1_feature_selected.get("refit_window_indices", [])]
                        if isinstance(phase1_feature_selected, dict)
                        else [int(i) for i in reopt_active_indices]
                    ),
                    "compile_cost_proxy": (
                        dict(phase1_feature_selected.get("compiled_position_cost_proxy", {}))
                        if isinstance(phase1_feature_selected, dict)
                        else None
                    ),
                    "measurement_cache_stats": (
                        dict(phase1_feature_selected.get("measurement_cache_stats", {}))
                        if isinstance(phase1_feature_selected, dict)
                        else None
                    ),
                    "actual_fallback_mode": (
                        str(phase1_feature_selected.get("actual_fallback_mode"))
                        if isinstance(phase1_feature_selected, dict)
                        else None
                    ),
                    "trough_probe_triggered": bool(phase1_last_trough_probe_triggered),
                    "trough_detected": bool(phase1_last_trough_detected),
                }
            )
            if phase2_enabled:
                history_row.update(
                    {
                        "full_v2_score": (
                            float(phase1_feature_selected.get("full_v2_score"))
                            if isinstance(phase1_feature_selected, dict)
                            and phase1_feature_selected.get("full_v2_score") is not None
                            else None
                        ),
                        "shortlist_size": int(len(phase2_last_shortlist_records)),
                        "shortlisted_records": [dict(x) for x in phase2_last_shortlist_records],
                        "compatibility_penalty_total": float(phase2_last_batch_penalty_total),
                        "optimizer_memory_reused": bool(phase2_last_optimizer_memory_reused),
                        "optimizer_memory_source": str(phase2_last_optimizer_memory_source),
                    }
                )
            if phase3_enabled:
                history_row.update(
                    {
                        "generator_id": (
                            str(phase1_feature_selected.get("generator_id"))
                            if isinstance(phase1_feature_selected, dict)
                            and phase1_feature_selected.get("generator_id") is not None
                            else None
                        ),
                        "template_id": (
                            str(phase1_feature_selected.get("template_id"))
                            if isinstance(phase1_feature_selected, dict)
                            and phase1_feature_selected.get("template_id") is not None
                            else None
                        ),
                        "is_macro_generator": (
                            bool(phase1_feature_selected.get("is_macro_generator", False))
                            if isinstance(phase1_feature_selected, dict)
                            else None
                        ),
                        "parent_generator_id": (
                            str(phase1_feature_selected.get("parent_generator_id"))
                            if isinstance(phase1_feature_selected, dict)
                            and phase1_feature_selected.get("parent_generator_id") is not None
                            else None
                        ),
                        "symmetry_mode": (
                            str(phase1_feature_selected.get("symmetry_mode"))
                            if isinstance(phase1_feature_selected, dict)
                            else None
                        ),
                        "symmetry_mitigation_mode": (
                            str(phase1_feature_selected.get("symmetry_mitigation_mode"))
                            if isinstance(phase1_feature_selected, dict)
                            else str(phase3_symmetry_mitigation_mode_key)
                        ),
                        "symmetry_spec": (
                            dict(phase1_feature_selected.get("symmetry_spec", {}))
                            if isinstance(phase1_feature_selected, dict)
                            else None
                        ),
                        "motif_bonus": (
                            float(phase1_feature_selected.get("motif_bonus", 0.0))
                            if isinstance(phase1_feature_selected, dict)
                            else 0.0
                        ),
                        "motif_source": (
                            str(phase1_feature_selected.get("motif_source"))
                            if isinstance(phase1_feature_selected, dict)
                            else None
                        ),
                        "motif_metadata": (
                            dict(phase1_feature_selected.get("motif_metadata", {}))
                            if isinstance(phase1_feature_selected, dict)
                            and isinstance(phase1_feature_selected.get("motif_metadata"), Mapping)
                            else None
                        ),
                        "runtime_split_mode": (
                            str(phase1_feature_selected.get("runtime_split_mode", "off"))
                            if isinstance(phase1_feature_selected, dict)
                            else "off"
                        ),
                        "runtime_split_parent_label": (
                            str(phase1_feature_selected.get("runtime_split_parent_label"))
                            if isinstance(phase1_feature_selected, dict)
                            and phase1_feature_selected.get("runtime_split_parent_label") is not None
                            else None
                        ),
                        "runtime_split_child_index": (
                            int(phase1_feature_selected.get("runtime_split_child_index"))
                            if isinstance(phase1_feature_selected, dict)
                            and phase1_feature_selected.get("runtime_split_child_index") is not None
                            else None
                        ),
                        "runtime_split_child_count": (
                            int(phase1_feature_selected.get("runtime_split_child_count"))
                            if isinstance(phase1_feature_selected, dict)
                            and phase1_feature_selected.get("runtime_split_child_count") is not None
                            else None
                        ),
                        "lifetime_cost_mode": (
                            str(phase1_feature_selected.get("lifetime_cost_mode"))
                            if isinstance(phase1_feature_selected, dict)
                            else str(phase3_lifetime_cost_mode_key)
                        ),
                        "remaining_evaluations_proxy_mode": (
                            str(phase1_feature_selected.get("remaining_evaluations_proxy_mode"))
                            if isinstance(phase1_feature_selected, dict)
                            else None
                        ),
                        "remaining_evaluations_proxy": (
                            float(phase1_feature_selected.get("remaining_evaluations_proxy", 0.0))
                            if isinstance(phase1_feature_selected, dict)
                            else 0.0
                        ),
                        "lifetime_weight_components": (
                            dict(phase1_feature_selected.get("lifetime_weight_components", {}))
                            if isinstance(phase1_feature_selected, dict)
                            else None
                        ),
                    }
                )
        if adapt_inner_optimizer_key == "SPSA":
            history_row["spsa_params"] = dict(adapt_spsa_params)
        history.append(history_row)
        if phase1_enabled:
            if selected_batch_records_for_history:
                for rec in selected_batch_records_for_history:
                    phase1_features_history.append(dict(rec))
            elif isinstance(phase1_feature_selected, dict):
                phase1_features_history.append(dict(phase1_feature_selected))
            phase1_measure_cache.commit([str(x) for x in selected_batch_labels] if selected_batch_labels else [str(pool[best_idx].label)])

        _ai_log(
            "hardcoded_adapt_iter_done",
            depth=int(depth + 1),
            energy=float(energy_current),
            delta_e=float(energy_current - energy_prev),
            eps_energy_step_abs=float(eps_energy_step_abs),
            eps_energy_low_step=bool(eps_energy_low_step),
            eps_energy_low_streak=int(eps_energy_low_streak),
            eps_energy_gate_open=bool(eps_energy_gate_open),
            eps_energy_min_extra_depth_effective=int(eps_energy_min_extra_depth_effective),
            eps_energy_patience_effective=int(eps_energy_patience_effective),
            eps_energy_termination_enabled=bool(eps_energy_termination_enabled),
            eps_energy_termination_condition=bool(eps_energy_termination_condition),
            eps_grad_termination_enabled=bool(eps_grad_termination_enabled),
            eps_grad_threshold_hit=bool(max_grad < float(eps_grad)),
            depth_cumulative=int(depth_cumulative),
            delta_abs_current=float(delta_abs_current),
            delta_abs_drop_from_prev=float(delta_abs_drop),
            drop_plateau_hits=int(drop_plateau_hits),
            depth_rollback=bool(depth_rollback),
            gradient_eval_elapsed_s=float(gradient_eval_elapsed_s),
            optimizer_elapsed_s=float(optimizer_elapsed_s),
        )

        if (
            drop_policy_enabled
            and int(depth_local) >= int(adapt_drop_min_depth)
            and int(drop_plateau_hits) >= int(adapt_drop_patience)
        ):
            if phase1_enabled and (not phase1_residual_opened) and len(phase1_residual_indices) > 0:
                phase1_residual_opened = True
                available_indices |= set(int(i) for i in phase1_residual_indices)
                phase1_stage.resolve_stage_transition(
                    drop_plateau_hits=int(drop_plateau_hits),
                    trough_detected=bool(phase1_last_trough_detected),
                    residual_opened=True,
                )
                drop_plateau_hits = 0
                phase1_stage_events.append(
                    {
                        "depth": int(depth_local),
                        "stage_name": "residual",
                        "reason": "drop_plateau_open",
                    }
                )
                _ai_log(
                    "hardcoded_adapt_phase1_residual_opened_on_plateau",
                    depth=int(depth_local),
                    residual_count=int(len(phase1_residual_indices)),
                )
                continue
            stop_reason = "drop_plateau"
            _ai_log(
                "hardcoded_adapt_converged_drop_plateau",
                depth=int(depth_local),
                delta_abs_current=float(delta_abs_current),
                delta_abs_drop_from_prev=float(delta_abs_drop),
                drop_floor=float(adapt_drop_floor),
                drop_patience=int(adapt_drop_patience),
                drop_min_depth=int(adapt_drop_min_depth),
                drop_plateau_hits=int(drop_plateau_hits),
                grad_floor=(float(adapt_grad_floor) if float(adapt_grad_floor) >= 0.0 else None),
                max_grad=float(max_grad),
            )
            break

        # 6) Check energy convergence
        if bool(eps_energy_termination_condition):
            if bool(eps_energy_termination_enabled):
                stop_reason = "eps_energy"
                _ai_log(
                    "hardcoded_adapt_converged_energy",
                    depth=int(depth_local),
                    depth_cumulative=int(depth_cumulative),
                    delta_e=float(eps_energy_step_abs),
                    eps_energy=float(eps_energy),
                    eps_energy_low_streak=int(eps_energy_low_streak),
                    eps_energy_patience=int(eps_energy_patience_effective),
                    eps_energy_min_extra_depth=int(eps_energy_min_extra_depth_effective),
                    adapt_ref_base_depth=int(adapt_ref_base_depth),
                    eps_energy_gate_cumulative_depth=int(adapt_ref_base_depth) + int(eps_energy_min_extra_depth_effective),
                )
                break
            _ai_log(
                "hardcoded_adapt_eps_energy_termination_suppressed",
                depth=int(depth_local),
                depth_cumulative=int(depth_cumulative),
                delta_e=float(eps_energy_step_abs),
                eps_energy=float(eps_energy),
                eps_energy_low_streak=int(eps_energy_low_streak),
                eps_energy_patience=int(eps_energy_patience_effective),
                eps_energy_min_extra_depth=int(eps_energy_min_extra_depth_effective),
                adapt_ref_base_depth=int(adapt_ref_base_depth),
                eps_energy_gate_cumulative_depth=int(adapt_ref_base_depth) + int(eps_energy_min_extra_depth_effective),
                continuation_mode=str(continuation_mode),
                problem=str(problem_key),
            )
        if bool(eps_energy_low_step) and (not bool(eps_energy_gate_open)):
            _ai_log(
                "hardcoded_adapt_energy_convergence_gate_wait",
                depth=int(depth_local),
                depth_cumulative=int(depth_cumulative),
                delta_e=float(eps_energy_step_abs),
                eps_energy=float(eps_energy),
                eps_energy_min_extra_depth=int(eps_energy_min_extra_depth_effective),
                adapt_ref_base_depth=int(adapt_ref_base_depth),
                eps_energy_gate_cumulative_depth=int(adapt_ref_base_depth) + int(eps_energy_min_extra_depth_effective),
            )

        # Check if pool exhausted
        if not allow_repeats and not available_indices:
            stop_reason = "pool_exhausted"
            _ai_log("hardcoded_adapt_pool_exhausted")
            break

    # -- Final full-prefix refit (windowed policy only) --
    final_full_refit_meta: dict[str, Any] = {
        "requested": bool(
            adapt_reopt_policy_key == "windowed" and adapt_final_full_refit_val
        ),
        "executed": False,
        "skipped_reason": None,
        "energy_before": None,
        "energy_after": None,
        "nfev": 0,
        "nit": 0,
        "opt_success": None,
        "opt_message": None,
        "rollback": False,
    }
    if (
        adapt_reopt_policy_key == "windowed"
        and adapt_final_full_refit_val
        and len(selected_ops) > 0
    ):
        # Check if already satisfied by last depth being a full-prefix reopt
        last_was_full = bool(
            len(history) > 0
            and str(history[-1].get("reopt_policy_effective", "")).startswith("windowed_periodic_full")
            or (len(history) > 0 and str(history[-1].get("reopt_policy_effective", "")) == "full")
        )
        # Also skip if last depth used all local indices (full or periodic full)
        if last_was_full and len(history) > 0:
            last_active = history[-1].get("reopt_active_count", 0)
            last_was_full = bool(int(last_active) == int(len(selected_ops)))

        if last_was_full:
            final_full_refit_meta["skipped_reason"] = "last_depth_already_full_prefix"
            _ai_log(
                "hardcoded_adapt_final_full_refit_skipped",
                reason="last_depth_already_full_prefix",
            )
        else:
            _ai_log(
                "hardcoded_adapt_final_full_refit_start",
                n_params=int(theta.size),
                energy_before=float(energy_current),
            )
            final_full_refit_meta["energy_before"] = float(energy_current)
            energy_before_final = float(energy_current)

            # Re-use existing _obj and optimizer infrastructure
            if adapt_state_backend_key == "compiled":
                if selected_executor is None:
                    selected_executor = _build_compiled_executor(selected_ops)

            # Reset COBYLA heartbeat state for final refit
            final_opt_t0 = time.perf_counter()
            cobyla_last_hb_t = final_opt_t0
            cobyla_nfev_so_far = 0
            cobyla_best_fun = float("inf")

            def _obj_final(x: np.ndarray) -> float:
                nonlocal cobyla_last_hb_t, cobyla_nfev_so_far, cobyla_best_fun
                if adapt_state_backend_key == "compiled":
                    assert selected_executor is not None
                    psi_obj = selected_executor.prepare_state(np.asarray(x, dtype=float), psi_ref)
                    energy_obj, _ = energy_via_one_apply(psi_obj, h_compiled)
                    energy_obj_val = float(energy_obj)
                else:
                    energy_obj_val = _adapt_energy_fn(
                        h_poly, psi_ref, selected_ops, x, h_compiled=h_compiled,
                    )
                if adapt_inner_optimizer_key == "COBYLA":
                    cobyla_nfev_so_far += 1
                    if energy_obj_val < cobyla_best_fun:
                        cobyla_best_fun = float(energy_obj_val)
                    now = time.perf_counter()
                    if (now - cobyla_last_hb_t) >= float(adapt_spsa_progress_every_s):
                        _ai_log(
                            "hardcoded_adapt_cobyla_heartbeat",
                            stage="final_full_refit",
                            nfev_opt_so_far=int(cobyla_nfev_so_far),
                            best_fun=float(cobyla_best_fun),
                            elapsed_opt_s=float(now - final_opt_t0),
                        )
                        cobyla_last_hb_t = now
                return float(energy_obj_val)

            final_x0 = np.array(theta, copy=True)
            if adapt_inner_optimizer_key == "SPSA":
                final_memory = None
                if phase2_enabled:
                    final_memory = phase2_memory_adapter.select_active(
                        phase2_optimizer_memory,
                        active_indices=list(range(int(theta.size))),
                        source="adapt.final_full_refit.active_subset",
                    )
                final_result = spsa_minimize(
                    fun=_obj_final,
                    x0=final_x0,
                    maxiter=int(maxiter),
                    seed=int(seed) + int(max_depth) + 1,
                    a=float(adapt_spsa_a),
                    c=float(adapt_spsa_c),
                    alpha=float(adapt_spsa_alpha),
                    gamma=float(adapt_spsa_gamma),
                    A=float(adapt_spsa_A),
                    bounds=None,
                    project="none",
                    eval_repeats=int(adapt_spsa_eval_repeats),
                    eval_agg=str(adapt_spsa_eval_agg_key),
                    avg_last=int(adapt_spsa_avg_last),
                    memory=(dict(final_memory) if isinstance(final_memory, Mapping) else None),
                    refresh_every=0,
                    precondition_mode=("diag_rms_grad" if phase2_enabled else "none"),
                )
            else:
                if scipy_minimize is None:
                    raise RuntimeError("SciPy minimize is unavailable for COBYLA final full refit.")
                final_result = scipy_minimize(
                    _obj_final,
                    final_x0,
                    method="COBYLA",
                    options={"maxiter": int(maxiter), "rhobeg": 0.3},
                )

            final_energy = float(final_result.fun)
            final_nfev = int(getattr(final_result, "nfev", 0))
            final_nit = int(getattr(final_result, "nit", 0))
            final_success = bool(getattr(final_result, "success", False))
            final_message = str(getattr(final_result, "message", ""))

            # Rollback-on-regression semantics
            if final_energy > energy_before_final:
                final_full_refit_meta["rollback"] = True
                _ai_log(
                    "hardcoded_adapt_final_full_refit_rollback",
                    energy_before=float(energy_before_final),
                    energy_after=float(final_energy),
                )
            else:
                theta = np.asarray(final_result.x, dtype=float)
                energy_current = float(final_energy)
                if phase2_enabled and adapt_inner_optimizer_key == "SPSA":
                    phase2_optimizer_memory = phase2_memory_adapter.merge_active(
                        phase2_optimizer_memory,
                        active_indices=list(range(int(theta.size))),
                        active_state=phase2_memory_adapter.from_result(
                            final_result,
                            method=str(adapt_inner_optimizer_key),
                            parameter_count=int(theta.size),
                            source="adapt.final_full_refit.result",
                        ),
                        source="adapt.final_full_refit.merge",
                    )

            nfev_total += final_nfev
            final_full_refit_meta["executed"] = True
            final_full_refit_meta["energy_after"] = float(final_energy)
            final_full_refit_meta["nfev"] = int(final_nfev)
            final_full_refit_meta["nit"] = int(final_nit)
            final_full_refit_meta["opt_success"] = bool(final_success)
            final_full_refit_meta["opt_message"] = str(final_message)
            _ai_log(
                "hardcoded_adapt_final_full_refit_done",
                energy_before=float(energy_before_final),
                energy_after=float(final_energy),
                rollback=bool(final_full_refit_meta["rollback"]),
                nfev=int(final_nfev),
            )

    prune_summary: dict[str, Any] = {
        "enabled": bool(phase1_enabled and phase1_prune_enabled),
        "executed": False,
        "rolled_back": False,
        "accepted_count": 0,
        "candidate_count": 0,
        "decisions": [],
        "energy_before": float(energy_current),
        "energy_after_prune": float(energy_current),
        "energy_after_post_refit": float(energy_current),
        "post_refit_executed": False,
    }
    if phase1_enabled and bool(phase1_prune_enabled) and int(len(selected_ops)) > 1:
        phase1_scaffold_pre_prune = {
            "operators": [str(op.label) for op in selected_ops],
            "optimal_point": [float(x) for x in np.asarray(theta, dtype=float).tolist()],
            "energy": float(energy_current),
        }
        prune_cfg = PruneConfig(
            max_candidates=int(max(1, phase1_prune_max_candidates)),
            min_candidates=2,
            fraction_candidates=float(max(0.0, phase1_prune_fraction)),
            max_regression=float(max(0.0, phase1_prune_max_regression)),
        )

        def _reconstruct_phase1_proxy_benefits() -> list[float]:
            benefits: list[float] = []
            for row in history:
                if not isinstance(row, dict):
                    continue
                if row.get("continuation_mode") not in {"phase1_v1", "phase2_v1", "phase3_v1"}:
                    continue
                pos = int(row.get("selected_position", len(benefits)))
                pos = max(0, min(len(benefits), pos))
                benefit = row.get("full_v2_score", row.get("simple_score", None))
                if benefit is None:
                    benefit = row.get("metric_proxy", row.get("selected_grad_abs", float("inf")))
                benefit_f = float(benefit)
                if not math.isfinite(benefit_f):
                    benefit_f = float("inf")
                benefits.insert(pos, benefit_f)
            while len(benefits) < int(len(selected_ops)):
                benefits.append(float("inf"))
            return [float(x) for x in benefits[: int(len(selected_ops))]]

        pre_prune_ops = list(selected_ops)
        pre_prune_theta = np.asarray(theta, dtype=float).copy()
        pre_prune_energy = float(energy_current)
        pre_prune_memory = dict(phase2_optimizer_memory) if phase2_enabled else None
        pre_prune_generator_meta = (
            selected_generator_metadata_for_labels(
                [str(op.label) for op in pre_prune_ops],
                pool_generator_registry,
            )
            if phase3_enabled
            else []
        )
        prune_proxy_benefit = _reconstruct_phase1_proxy_benefits()
        candidate_indices = rank_prune_candidates(
            theta=np.asarray(theta, dtype=float),
            labels=[str(op.label) for op in selected_ops],
            marginal_proxy_benefit=list(prune_proxy_benefit),
            max_candidates=int(prune_cfg.max_candidates),
            min_candidates=int(prune_cfg.min_candidates),
            fraction_candidates=float(prune_cfg.fraction_candidates),
        )
        prune_summary["candidate_count"] = int(len(candidate_indices))
        prune_summary["marginal_proxy_benefit"] = [float(x) for x in prune_proxy_benefit]

        def _refit_given_ops(ops_refit: list[AnsatzTerm], theta0: np.ndarray) -> tuple[np.ndarray, float]:
            nonlocal phase2_optimizer_memory
            if len(ops_refit) == 0:
                return np.zeros(0, dtype=float), float(energy_current)
            executor_refit = _build_compiled_executor(ops_refit) if adapt_state_backend_key == "compiled" else None

            def _obj_prune(x: np.ndarray) -> float:
                if executor_refit is not None:
                    psi_obj = executor_refit.prepare_state(np.asarray(x, dtype=float), psi_ref)
                    e_obj, _ = energy_via_one_apply(psi_obj, h_compiled)
                    return float(e_obj)
                return float(_adapt_energy_fn(h_poly, psi_ref, ops_refit, x, h_compiled=h_compiled))

            x0 = np.asarray(theta0, dtype=float).reshape(-1)
            if adapt_inner_optimizer_key == "SPSA":
                refit_memory = None
                if phase2_enabled:
                    refit_memory = phase2_memory_adapter.select_active(
                        phase2_optimizer_memory,
                        active_indices=list(range(int(x0.size))),
                        source="adapt.post_prune_refit.active_subset",
                    )
                res = spsa_minimize(
                    fun=_obj_prune,
                    x0=x0,
                    maxiter=int(max(25, min(int(maxiter), 120))),
                    seed=int(seed) + 700000 + int(len(ops_refit)),
                    a=float(adapt_spsa_a),
                    c=float(adapt_spsa_c),
                    alpha=float(adapt_spsa_alpha),
                    gamma=float(adapt_spsa_gamma),
                    A=float(adapt_spsa_A),
                    bounds=None,
                    project="none",
                    eval_repeats=int(adapt_spsa_eval_repeats),
                    eval_agg=str(adapt_spsa_eval_agg_key),
                    avg_last=int(adapt_spsa_avg_last),
                    memory=(dict(refit_memory) if isinstance(refit_memory, Mapping) else None),
                    refresh_every=0,
                    precondition_mode=("diag_rms_grad" if phase2_enabled else "none"),
                )
                if phase2_enabled:
                    phase2_optimizer_memory = phase2_memory_adapter.merge_active(
                        phase2_optimizer_memory,
                        active_indices=list(range(int(x0.size))),
                        active_state=phase2_memory_adapter.from_result(
                            res,
                            method=str(adapt_inner_optimizer_key),
                            parameter_count=int(x0.size),
                            source="adapt.post_prune_refit.result",
                        ),
                        source="adapt.post_prune_refit.merge",
                    )
                return np.asarray(res.x, dtype=float), float(res.fun)
            if scipy_minimize is None:
                raise RuntimeError("SciPy minimize is unavailable for prune refit.")
            res = scipy_minimize(
                _obj_prune,
                x0,
                method="COBYLA",
                options={"maxiter": int(max(25, min(int(maxiter), 120))), "rhobeg": 0.3},
            )
            return np.asarray(res.x, dtype=float), float(res.fun)

        def _ops_from_labels(labels_cur: list[str]) -> list[AnsatzTerm]:
            buckets: dict[str, list[AnsatzTerm]] = {}
            for op_ref in selected_ops:
                key_ref = str(op_ref.label)
                buckets.setdefault(key_ref, []).append(op_ref)
            rebuilt: list[AnsatzTerm] = []
            for lbl in labels_cur:
                key_lbl = str(lbl)
                if key_lbl not in buckets or len(buckets[key_lbl]) == 0:
                    continue
                rebuilt.append(buckets[key_lbl].pop(0))
            return rebuilt

        def _eval_with_removal(
            idx_remove: int,
            theta_cur: np.ndarray,
            labels_cur: list[str],
        ) -> tuple[float, np.ndarray]:
            ops_trial = _ops_from_labels(list(labels_cur))
            del ops_trial[int(idx_remove)]
            theta_trial0 = np.delete(np.asarray(theta_cur, dtype=float), int(idx_remove))
            theta_trial_opt, e_trial = _refit_given_ops(ops_trial, theta_trial0)
            return float(e_trial), np.asarray(theta_trial_opt, dtype=float)

        theta_pruned, labels_pruned, prune_decisions, energy_after_prune = apply_pruning(
            theta=np.asarray(theta, dtype=float),
            labels=[str(op.label) for op in selected_ops],
            candidate_indices=[int(i) for i in candidate_indices],
            eval_with_removal=_eval_with_removal,
            energy_before=float(energy_current),
            max_regression=float(prune_cfg.max_regression),
        )
        accepted_count = int(sum(1 for d in prune_decisions if bool(d.accepted)))
        if accepted_count > 0:
            accepted_remove_indices = [int(d.index) for d in prune_decisions if bool(d.accepted)]
            if phase2_enabled:
                phase2_optimizer_memory = phase2_memory_adapter.remap_remove(
                    phase2_optimizer_memory,
                    indices=list(accepted_remove_indices),
                )
            label_to_ops: dict[str, list[AnsatzTerm]] = {}
            for op in selected_ops:
                key = str(op.label)
                label_to_ops.setdefault(key, []).append(op)
            rebuilt_ops: list[AnsatzTerm] = []
            for lbl in labels_pruned:
                key = str(lbl)
                bucket = label_to_ops.get(key, [])
                if not bucket:
                    continue
                rebuilt_ops.append(bucket.pop(0))
            selected_ops = list(rebuilt_ops)
            theta = np.asarray(theta_pruned, dtype=float)
            energy_current = float(energy_after_prune)
            if adapt_state_backend_key == "compiled":
                selected_executor = _build_compiled_executor(selected_ops) if len(selected_ops) > 0 else None
            else:
                selected_executor = None
            theta_post, e_post = post_prune_refit(
                theta=np.asarray(theta, dtype=float),
                refit_fn=lambda x: _refit_given_ops(list(selected_ops), np.asarray(x, dtype=float)),
            )
            theta = np.asarray(theta_post, dtype=float)
            energy_current = float(e_post)
            if adapt_state_backend_key == "compiled":
                selected_executor = _build_compiled_executor(selected_ops) if len(selected_ops) > 0 else None
            else:
                selected_executor = None
            prune_summary["post_refit_executed"] = True
            if phase3_enabled and str(phase3_symmetry_mitigation_mode_key) != "off":
                post_prune_generator_meta = selected_generator_metadata_for_labels(
                    [str(op.label) for op in selected_ops],
                    pool_generator_registry,
                )
                sym_pre = verify_symmetry_sequence(
                    generator_metadata=pre_prune_generator_meta,
                    mitigation_mode=str(phase3_symmetry_mitigation_mode_key),
                )
                sym_post = verify_symmetry_sequence(
                    generator_metadata=post_prune_generator_meta,
                    mitigation_mode=str(phase3_symmetry_mitigation_mode_key),
                )
                prune_summary["symmetry_mitigation"] = {
                    "mode": str(phase3_symmetry_mitigation_mode_key),
                    "pre_prune": dict(sym_pre),
                    "post_prune": dict(sym_post),
                }
                if (not bool(sym_post.get("passed", True))) or (
                    float(sym_post.get("max_leakage_risk", 0.0))
                    > float(sym_pre.get("max_leakage_risk", 0.0)) + 1e-12
                ):
                    selected_ops = list(pre_prune_ops)
                    theta = np.asarray(pre_prune_theta, dtype=float)
                    energy_current = float(pre_prune_energy)
                    if phase2_enabled and isinstance(pre_prune_memory, dict):
                        phase2_optimizer_memory = dict(pre_prune_memory)
                    if adapt_state_backend_key == "compiled":
                        selected_executor = _build_compiled_executor(selected_ops) if len(selected_ops) > 0 else None
                    else:
                        selected_executor = None
                    prune_summary["rolled_back"] = True
                    prune_summary["rollback_reason"] = "symmetry_verify_failed"
            if float(energy_current) > float(pre_prune_energy) + float(prune_cfg.max_regression):
                selected_ops = list(pre_prune_ops)
                theta = np.asarray(pre_prune_theta, dtype=float)
                energy_current = float(pre_prune_energy)
                if phase2_enabled and isinstance(pre_prune_memory, dict):
                    phase2_optimizer_memory = dict(pre_prune_memory)
                if adapt_state_backend_key == "compiled":
                    selected_executor = _build_compiled_executor(selected_ops) if len(selected_ops) > 0 else None
                else:
                    selected_executor = None
                prune_summary["rolled_back"] = True
                prune_summary["rollback_reason"] = "post_prune_regression_exceeded"

        prune_summary.update(
            {
                "executed": True,
                "accepted_count": int(accepted_count),
                "energy_after_prune": float(energy_after_prune),
                "energy_after_post_refit": float(energy_current),
                "decisions": [dict(d.__dict__) for d in prune_decisions],
            }
        )
        _ai_log(
            "hardcoded_adapt_phase1_prune_done",
            candidate_count=int(prune_summary["candidate_count"]),
            accepted_count=int(prune_summary["accepted_count"]),
            energy_after_post_refit=float(prune_summary["energy_after_post_refit"]),
        )

    # Build final state
    if adapt_state_backend_key == "compiled":
        if len(selected_ops) == 0:
            psi_adapt = np.array(psi_ref, copy=True)
        else:
            if selected_executor is None:
                selected_executor = _build_compiled_executor(selected_ops)
            psi_adapt = selected_executor.prepare_state(theta, psi_ref)
    else:
        psi_adapt = _prepare_adapt_state(psi_ref, selected_ops, theta)
    psi_adapt = _normalize_state(psi_adapt)

    elapsed = time.perf_counter() - t0
    selected_generator_metadata = (
        selected_generator_metadata_for_labels(
            [str(op.label) for op in selected_ops],
            pool_generator_registry,
        )
        if phase3_enabled
        else []
    )
    phase3_symmetry_summary = (
        verify_symmetry_sequence(
            generator_metadata=selected_generator_metadata,
            mitigation_mode=str(phase3_symmetry_mitigation_mode_key),
        )
        if phase3_enabled
        else None
    )
    phase3_output_motif_library = (
        extract_motif_library(
            generator_metadata=selected_generator_metadata,
            theta=[float(x) for x in np.asarray(theta, dtype=float).tolist()],
            source_num_sites=int(num_sites),
            source_tag=f"phase3_v1_L{int(num_sites)}",
            ordering=str(ordering),
            boson_encoding=str(boson_encoding),
        )
        if phase3_enabled and selected_generator_metadata
        else None
    )
    if phase3_enabled and isinstance(phase3_input_motif_library, Mapping):
        selected_match_count = 0
        source_records = phase3_input_motif_library.get("records", [])
        if isinstance(source_records, Sequence):
            for meta in selected_generator_metadata:
                matched = False
                for rec in source_records:
                    if not isinstance(rec, Mapping):
                        continue
                    if str(rec.get("family_id", "")) != str(meta.get("family_id", "")):
                        continue
                    if str(rec.get("template_id", "")) != str(meta.get("template_id", "")):
                        continue
                    if [int(x) for x in rec.get("support_site_offsets", [])] != [
                        int(x) for x in meta.get("support_site_offsets", [])
                    ]:
                        continue
                    matched = True
                    break
                if matched:
                    selected_match_count += 1
        phase3_motif_usage["selected_match_count"] = int(selected_match_count)

    continuation_payload: dict[str, Any] = {
        "mode": str(continuation_mode),
        "score_version": (
            str(phase2_score_cfg.score_version)
            if phase2_enabled
            else str(phase1_score_cfg.score_version)
        ),
        "stage_controller": {
            "plateau_patience": int(phase1_stage_cfg.plateau_patience),
            "probe_margin_ratio": float(phase1_stage_cfg.probe_margin_ratio),
            "max_probe_positions": int(phase1_stage_cfg.max_probe_positions),
            "append_admit_threshold": float(phase1_stage_cfg.append_admit_threshold),
        },
        "stage_events": [dict(row) for row in phase1_stage_events],
        "phase1_feature_rows": [dict(row) for row in phase1_features_history[-200:]],
        "last_probe_reason": str(phase1_last_probe_reason),
        "residual_opened": bool(phase1_residual_opened),
    }
    if phase2_enabled:
        continuation_payload.update(
            {
                "phase2_shortlist_rows": [dict(row) for row in phase2_last_shortlist_records[-200:]],
                "optimizer_memory": dict(phase2_optimizer_memory),
                "phase2": {
                    "shortlist_fraction": float(phase2_score_cfg.shortlist_fraction),
                    "shortlist_size": int(phase2_score_cfg.shortlist_size),
                    "batch_target_size": int(phase2_score_cfg.batch_target_size),
                    "batch_size_cap": int(phase2_score_cfg.batch_size_cap),
                    "batch_near_degenerate_ratio": float(phase2_score_cfg.batch_near_degenerate_ratio),
                },
            }
        )
    if phase3_enabled:
        continuation_payload.update(
            {
                "selected_generator_metadata": [dict(x) for x in selected_generator_metadata],
                "generator_split_events": [dict(x) for x in phase3_split_events],
                "runtime_split_summary": dict(phase3_runtime_split_summary),
                "motif_library": (
                    dict(phase3_output_motif_library)
                    if isinstance(phase3_output_motif_library, Mapping)
                    else None
                ),
                "motif_usage": dict(phase3_motif_usage),
                "symmetry_mitigation": (
                    dict(phase3_symmetry_summary)
                    if isinstance(phase3_symmetry_summary, Mapping)
                    else None
                ),
                "rescue_history": [dict(x) for x in phase3_rescue_history],
            }
        )

    payload = {
        "success": True,
        "method": method_name,
        "energy": float(energy_current),
        "exact_gs_energy": float(exact_gs),
        "delta_e": float(energy_current - exact_gs),
        "abs_delta_e": float(abs(energy_current - exact_gs)),
        "num_particles": {"n_up": int(num_particles[0]), "n_dn": int(num_particles[1])},
        "ansatz_depth": int(len(selected_ops)),
        "num_parameters": int(theta.size),
        "optimal_point": [float(x) for x in theta.tolist()],
        "operators": [str(op.label) for op in selected_ops],
        "pool_size": int(len(pool)),
        "pool_type": str(pool_key),
        "stop_reason": str(stop_reason),
        "nfev_total": int(nfev_total),
        "adapt_inner_optimizer": str(adapt_inner_optimizer_key),
        "adapt_reopt_policy": str(adapt_reopt_policy_key),
        "adapt_window_size": int(adapt_window_size_val),
        "adapt_window_topk": int(adapt_window_topk_val),
        "adapt_full_refit_every": int(adapt_full_refit_every_val),
        "adapt_final_full_refit": bool(adapt_final_full_refit_val),
        "allow_repeats": bool(allow_repeats),
        "finite_angle_fallback": bool(finite_angle_fallback),
        "finite_angle": float(finite_angle),
        "finite_angle_min_improvement": float(finite_angle_min_improvement),
        "adapt_drop_policy_enabled": bool(drop_policy_enabled),
        "adapt_drop_floor": (float(adapt_drop_floor) if drop_policy_enabled else None),
        "adapt_drop_patience": (int(adapt_drop_patience) if drop_policy_enabled else None),
        "adapt_drop_min_depth": (int(adapt_drop_min_depth) if drop_policy_enabled else None),
        "adapt_grad_floor": (float(adapt_grad_floor) if float(adapt_grad_floor) >= 0.0 else None),
        "adapt_drop_floor_resolved": float(adapt_drop_floor),
        "adapt_drop_patience_resolved": int(adapt_drop_patience),
        "adapt_drop_min_depth_resolved": int(adapt_drop_min_depth),
        "adapt_grad_floor_resolved": float(adapt_grad_floor),
        "adapt_drop_floor_source": str(stop_policy.adapt_drop_floor_source),
        "adapt_drop_patience_source": str(stop_policy.adapt_drop_patience_source),
        "adapt_drop_min_depth_source": str(stop_policy.adapt_drop_min_depth_source),
        "adapt_grad_floor_source": str(stop_policy.adapt_grad_floor_source),
        "adapt_drop_policy_source": str(stop_policy.drop_policy_source),
        "adapt_ref_base_depth": int(adapt_ref_base_depth),
        "adapt_eps_energy_min_extra_depth": int(adapt_eps_energy_min_extra_depth),
        "adapt_eps_energy_patience": int(adapt_eps_energy_patience),
        "eps_energy_min_extra_depth_effective": int(eps_energy_min_extra_depth_effective),
        "eps_energy_patience_effective": int(eps_energy_patience_effective),
        "eps_energy_gate_cumulative_depth": int(adapt_ref_base_depth) + int(eps_energy_min_extra_depth_effective),
        "eps_energy_termination_enabled": bool(eps_energy_termination_enabled),
        "eps_grad_termination_enabled": bool(eps_grad_termination_enabled),
        "eps_energy_low_streak_final": int(eps_energy_low_streak),
        "drop_plateau_hits_final": int(drop_plateau_hits),
        "adapt_gradient_parity_check": bool(adapt_gradient_parity_check),
        "adapt_state_backend": str(adapt_state_backend_key),
        "compiled_pauli_cache": {
            "enabled": True,
            "compile_elapsed_s": compile_cache_elapsed_s,
            "h_terms": int(len(h_compiled.terms)),
            "pool_terms_total": int(pool_compiled_terms_total),
            "unique_pauli_actions": int(len(pauli_action_cache)),
        },
        "history": history,
        "final_full_refit": dict(final_full_refit_meta),
        "continuation_mode": str(continuation_mode),
        "elapsed_s": float(elapsed),
        "hf_bitstring_qn_to_q0": str(hf_bits),
    }
    if phase1_enabled:
        measurement_plan = phase1_measure_cache.plan_for([])
        payload.update(
            {
                "continuation": dict(continuation_payload),
                "measurement_cache_summary": {
                    **dict(phase1_measure_cache.summary()),
                    "measurement_plan": dict(measurement_plan.__dict__),
                },
                "compile_cost_proxy_summary": {
                    "version": (
                        "phase3_v1_proxy"
                        if phase3_enabled
                        else ("phase2_v1_proxy" if phase2_enabled else "phase1_v1_proxy")
                    ),
                    "components": [
                        "new_pauli_actions",
                        "new_rotation_steps",
                        "position_shift_span",
                        "refit_active_count",
                    ],
                },
                "pre_prune_scaffold": (
                    dict(phase1_scaffold_pre_prune) if phase1_scaffold_pre_prune is not None else None
                ),
                "prune_summary": dict(prune_summary),
                "post_prune_refit": {
                    "executed": bool(prune_summary.get("post_refit_executed", False)),
                    "energy": float(prune_summary.get("energy_after_post_refit", energy_current)),
                },
                "scaffold_fingerprint_lite": ScaffoldFingerprintLite(
                    selected_operator_labels=[str(op.label) for op in selected_ops],
                    selected_generator_ids=[
                        str(meta.get("generator_id", ""))
                        for meta in selected_generator_metadata
                        if str(meta.get("generator_id", "")) != ""
                    ],
                    num_parameters=int(theta.size),
                    generator_family=str(pool_key),
                    continuation_mode=str(continuation_mode),
                    compiled_pauli_cache_size=int(len(pauli_action_cache)),
                    measurement_plan_version=str(measurement_plan.plan_version),
                    post_prune=bool(prune_summary.get("executed", False)),
                    split_event_count=int(len(phase3_split_events)),
                    motif_record_ids=(
                        [
                            str(rec.get("motif_id", ""))
                            for rec in phase3_output_motif_library.get("records", [])
                            if isinstance(rec, Mapping) and str(rec.get("motif_id", "")) != ""
                        ]
                        if isinstance(phase3_output_motif_library, Mapping)
                        else []
                    ),
                ).__dict__,
            }
        )
    if adapt_inner_optimizer_key == "SPSA":
        payload["adapt_spsa"] = dict(adapt_spsa_params)

    _ai_log(
        "hardcoded_adapt_vqe_done",
        L=int(num_sites),
        adapt_inner_optimizer=str(adapt_inner_optimizer_key),
        energy=float(energy_current),
        exact_gs=float(exact_gs),
        abs_delta_e=float(abs(energy_current - exact_gs)),
        depth=int(len(selected_ops)),
        stop_reason=str(stop_reason),
        elapsed_sec=round(elapsed, 6),
    )
    return payload, psi_adapt


# ---------------------------------------------------------------------------
# Trajectory simulation (identical to VQE pipeline)
# ---------------------------------------------------------------------------

def _simulate_trajectory(
    *,
    num_sites: int,
    psi0: np.ndarray,
    hmat: np.ndarray,
    ordered_labels_exyz: list[str],
    coeff_map_exyz: dict[str, complex],
    trotter_steps: int,
    t_final: float,
    num_times: int,
    suzuki_order: int,
) -> tuple[list[dict[str, float]], list[np.ndarray]]:
    if int(suzuki_order) != 2:
        raise ValueError("This script currently supports suzuki_order=2 only.")

    nq = len(ordered_labels_exyz[0]) if ordered_labels_exyz else int(np.log2(max(1, psi0.size)))
    evals, evecs = np.linalg.eigh(hmat)
    evecs_dag = np.conjugate(evecs).T

    compiled = {lbl: _compile_pauli_action(lbl, nq) for lbl in ordered_labels_exyz}
    times = np.linspace(0.0, float(t_final), int(num_times))
    n_times = int(times.size)
    stride = max(1, n_times // 20)
    t0 = time.perf_counter()
    _ai_log(
        "hardcoded_adapt_trajectory_start",
        L=int(num_sites),
        num_times=n_times,
        t_final=float(t_final),
        trotter_steps=int(trotter_steps),
    )

    rows: list[dict[str, float]] = []
    exact_states: list[np.ndarray] = []

    for idx, time_val in enumerate(times):
        tv = float(time_val)
        psi_exact = evecs @ (np.exp(-1j * evals * tv) * (evecs_dag @ psi0))
        psi_exact = _normalize_state(psi_exact)

        psi_trot = _evolve_trotter_suzuki2_absolute(
            psi0, ordered_labels_exyz, coeff_map_exyz, compiled, tv, int(trotter_steps),
        )

        fidelity = float(abs(np.vdot(psi_exact, psi_trot)) ** 2)
        n_up_exact, n_dn_exact = _occupation_site0(psi_exact, num_sites)
        n_up_trot, n_dn_trot = _occupation_site0(psi_trot, num_sites)

        rows.append({
            "time": tv,
            "fidelity": fidelity,
            "energy_exact": _expectation_hamiltonian(psi_exact, hmat),
            "energy_trotter": _expectation_hamiltonian(psi_trot, hmat),
            "n_up_site0_exact": n_up_exact,
            "n_up_site0_trotter": n_up_trot,
            "n_dn_site0_exact": n_dn_exact,
            "n_dn_site0_trotter": n_dn_trot,
            "doublon_exact": _doublon_total(psi_exact, num_sites),
            "doublon_trotter": _doublon_total(psi_trot, num_sites),
        })
        exact_states.append(psi_exact)
        if idx == 0 or idx == n_times - 1 or ((idx + 1) % stride == 0):
            _ai_log(
                "hardcoded_adapt_trajectory_progress",
                step=int(idx + 1),
                total_steps=n_times,
                frac=round(float((idx + 1) / n_times), 6),
                time=tv,
                fidelity=float(fidelity),
                elapsed_sec=round(time.perf_counter() - t0, 6),
            )

    _ai_log("hardcoded_adapt_trajectory_done", L=int(num_sites), num_times=n_times)
    return rows, exact_states


# ---------------------------------------------------------------------------
# PDF writer (compact — mirrors VQE pipeline)
# ---------------------------------------------------------------------------

def _write_pipeline_pdf(pdf_path: Path, payload: dict[str, Any], run_command: str) -> None:
    with PdfPages(str(pdf_path)) as pdf:
        # Parameter manifest (AGENTS.md requirement)
        settings = payload.get("settings", {})
        adapt = payload.get("adapt_vqe", {})
        problem = settings.get("problem", "hubbard")

        manifest_lines = [
            "ADAPT-VQE Pipeline — Parameter Manifest",
            "",
            f"Model:           {'Hubbard-Holstein' if problem == 'hh' else 'Hubbard'}",
            f"Ansatz type:     ADAPT-VQE (pool: {settings.get('adapt_pool', '?')})",
            f"Drive:           disabled (static ADAPT pipeline)",
            f"t = {settings.get('t')}    U = {settings.get('u')}    dv = {settings.get('dv')}",
            f"L = {settings.get('L')}    boundary = {settings.get('boundary')}    ordering = {settings.get('ordering')}",
        ]
        if problem == "hh":
            manifest_lines += [
                f"omega0 = {settings.get('omega0')}    g_ep = {settings.get('g_ep')}",
                f"n_ph_max = {settings.get('n_ph_max')}    boson_encoding = {settings.get('boson_encoding')}",
            ]
        manifest_lines += [
            "",
            f"ADAPT max depth:         {settings.get('adapt_max_depth', '?')}",
            f"ADAPT eps_grad:          {settings.get('adapt_eps_grad', '?')}",
            f"ADAPT eps_energy:        {settings.get('adapt_eps_energy', '?')}",
            f"ADAPT inner optimizer:   {settings.get('adapt_inner_optimizer', '?')}",
            f"ADAPT finite angle fb:   {settings.get('adapt_finite_angle_fallback', '?')}",
            f"ADAPT finite angle:      {settings.get('adapt_finite_angle', '?')}",
            "",
            f"Trotter steps:           {settings.get('trotter_steps')}",
            f"t_final:                 {settings.get('t_final')}",
            f"Suzuki order:            {settings.get('suzuki_order')}",
        ]
        if str(settings.get("adapt_inner_optimizer", "")).strip().upper() == "SPSA":
            adapt_spsa = settings.get("adapt_spsa", {})
            if isinstance(adapt_spsa, dict):
                manifest_lines += [
                    f"SPSA: a={adapt_spsa.get('a')}  c={adapt_spsa.get('c')}  A={adapt_spsa.get('A')}",
                    f"SPSA: alpha={adapt_spsa.get('alpha')}  gamma={adapt_spsa.get('gamma')}",
                    (
                        "SPSA: eval_repeats={eval_repeats}  eval_agg={eval_agg}  avg_last={avg_last}".format(
                            eval_repeats=adapt_spsa.get("eval_repeats"),
                            eval_agg=adapt_spsa.get("eval_agg"),
                            avg_last=adapt_spsa.get("avg_last"),
                        )
                    ),
                ]
        render_text_page(pdf, manifest_lines, fontsize=10, line_spacing=0.03)

        # Command page
        render_text_page(pdf, [
            "Executed Command",
            "",
            "Script: pipelines/hardcoded/adapt_pipeline.py",
            "",
            run_command,
        ], fontsize=10, line_spacing=0.03, max_line_width=110)

        # Settings + ADAPT summary page
        lines = [
            "Hardcoded ADAPT-VQE Pipeline Summary",
            "",
            f"L={settings.get('L')}  t={settings.get('t')}  u={settings.get('u')}  dv={settings.get('dv')}",
            f"boundary={settings.get('boundary')}  ordering={settings.get('ordering')}",
            f"initial_state_source={settings.get('initial_state_source')}",
            f"adapt_inner_optimizer={settings.get('adapt_inner_optimizer')}",
            "",
            f"ADAPT-VQE energy:  {adapt.get('energy')}",
            f"Exact GS energy:   {adapt.get('exact_gs_energy')}",
            f"|ΔE|:              {adapt.get('abs_delta_e')}",
            f"Ansatz depth:      {adapt.get('ansatz_depth')}",
            f"Pool size:         {adapt.get('pool_size')}",
            f"Stop reason:       {adapt.get('stop_reason')}",
            f"Total nfev:        {adapt.get('nfev_total')}",
            f"Elapsed:           {adapt.get('elapsed_s'):.2f}s" if adapt.get("elapsed_s") is not None else "",
            "",
            "Selected operators:",
        ]
        for op_label in (adapt.get("operators") or []):
            lines.append(f"  {op_label}")
        render_text_page(pdf, lines)

        # Trajectory plots
        rows = payload.get("trajectory", [])
        if rows:
            times = np.array([r["time"] for r in rows])
            fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), sharex=True)
            ax_f, ax_e = axes[0]
            ax_n, ax_d = axes[1]

            ax_f.plot(times, [r["fidelity"] for r in rows], color="#0b3d91")
            ax_f.set_title("Fidelity (Trotter vs Exact)")
            ax_f.set_ylabel("F(t)")
            ax_f.grid(alpha=0.25)

            ax_e.plot(times, [r["energy_trotter"] for r in rows], label="Trotter", color="#d62728")
            ax_e.plot(times, [r["energy_exact"] for r in rows], label="Exact", color="#111111", linestyle="--")
            ax_e.set_title("Energy")
            ax_e.set_ylabel("E(t)")
            ax_e.legend(fontsize=8)
            ax_e.grid(alpha=0.25)

            ax_n.plot(times, [r["n_up_site0_trotter"] for r in rows], label="n_up trot", color="#17becf")
            ax_n.plot(times, [r["n_dn_site0_trotter"] for r in rows], label="n_dn trot", color="#9467bd")
            ax_n.set_title("Site-0 Occupations (Trotter)")
            ax_n.set_xlabel("Time")
            ax_n.legend(fontsize=8)
            ax_n.grid(alpha=0.25)

            ax_d.plot(times, [r["doublon_trotter"] for r in rows], label="Trotter", color="#e377c2")
            ax_d.plot(times, [r["doublon_exact"] for r in rows], label="Exact", color="#111111", linestyle="--")
            ax_d.set_title("Doublon")
            ax_d.set_xlabel("Time")
            ax_d.legend(fontsize=8)
            ax_d.grid(alpha=0.25)

            fig.suptitle(f"Hardcoded ADAPT-VQE Pipeline L={settings.get('L')}", fontsize=13)
            fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
            pdf.savefig(fig)
            plt.close(fig)


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hardcoded ADAPT-VQE Hubbard / Hubbard-Holstein pipeline.")
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--u", type=float, default=4.0)
    p.add_argument("--problem", choices=["hubbard", "hh"], default="hubbard")
    p.add_argument("--dv", type=float, default=0.0)
    p.add_argument("--omega0", type=float, default=0.0)
    p.add_argument("--g-ep", type=float, default=0.0, help="Holstein electron-phonon coupling g.")
    p.add_argument("--n-ph-max", type=int, default=1)
    p.add_argument("--boson-encoding", choices=["binary"], default="binary")
    p.add_argument("--boundary", choices=["periodic", "open"], default="periodic")
    p.add_argument("--ordering", choices=["blocked", "interleaved"], default="blocked")
    p.add_argument("--term-order", choices=["native", "sorted"], default="sorted")

    # ADAPT-VQE controls
    p.add_argument(
        "--adapt-pool",
        choices=[
            "uccsd",
            "cse",
            "full_hamiltonian",
            "hva",
            "full_meta",
            "uccsd_paop_lf_full",
            "paop",
            "paop_min",
            "paop_std",
            "paop_full",
            "paop_lf",
            "paop_lf_std",
            "paop_lf2_std",
            "paop_lf_full",
        ],
        default=None,
        help=(
            "ADAPT pool family. If omitted, runtime resolves problem-aware defaults: "
            "hubbard->uccsd, hh+legacy->full_meta, hh+phase1_v1/phase2_v1/phase3_v1->paop_lf_std core + residual full_meta."
        ),
    )
    p.add_argument(
        "--adapt-continuation-mode",
        choices=["legacy", "phase1_v1", "phase2_v1", "phase3_v1"],
        default="legacy",
        help="Continuation mode for ADAPT. legacy is default; phase1_v1 is staged continuation; phase2_v1 adds shortlist/full scoring and batching; phase3_v1 adds generator/motif/symmetry/rescue metadata.",
    )
    p.add_argument("--adapt-max-depth", type=int, default=20)
    p.add_argument("--adapt-eps-grad", type=float, default=1e-4)
    p.add_argument(
        "--adapt-eps-energy",
        type=float,
        default=1e-8,
        help=(
            "Energy convergence threshold. Acts as a terminating guard for Hubbard and HH legacy runs; "
            "in HH phase1_v1/phase2_v1/phase3_v1 it is telemetry-only."
        ),
    )
    p.add_argument(
        "--adapt-inner-optimizer",
        choices=["COBYLA", "SPSA"],
        default="SPSA",
        help="Inner re-optimizer for HH seed pre-opt and per-depth ADAPT re-optimization.",
    )
    p.add_argument(
        "--adapt-state-backend",
        choices=["compiled", "legacy"],
        default="compiled",
        help="State action backend for ADAPT gradient/energy evaluations (compiled is default production path).",
    )
    p.add_argument(
        "--adapt-reopt-policy",
        choices=["append_only", "full", "windowed"],
        default="append_only",
        help=(
            "Per-depth re-optimization policy. "
            "'append_only' (default): freeze the prefix theta[:k] and optimize only the newly appended parameter. "
            "'full': legacy behavior — re-optimize all parameters jointly. "
            "'windowed': optimize a sliding window of recent parameters plus optional top-k older carry."
        ),
    )
    p.add_argument(
        "--adapt-window-size", type=int, default=3,
        help="Window size for 'windowed' reopt policy (number of newest parameters in active set).",
    )
    p.add_argument(
        "--adapt-window-topk", type=int, default=0,
        help="Number of older high-magnitude parameters to include in windowed active set.",
    )
    p.add_argument(
        "--adapt-full-refit-every", type=int, default=0,
        help="Periodic full-prefix refit cadence for 'windowed' (0=disabled). Uses cumulative depth.",
    )
    p.add_argument(
        "--adapt-final-full-refit",
        choices=["true", "false"],
        default="true",
        help="Run a final full-prefix refit after ADAPT loop when using 'windowed' policy.",
    )
    p.add_argument("--phase1-lambda-F", type=float, default=1.0)
    p.add_argument("--phase1-lambda-compile", type=float, default=0.05)
    p.add_argument("--phase1-lambda-measure", type=float, default=0.02)
    p.add_argument("--phase1-lambda-leak", type=float, default=0.0)
    p.add_argument("--phase1-score-z-alpha", type=float, default=0.0)
    p.add_argument("--phase1-probe-max-positions", type=int, default=6)
    p.add_argument("--phase1-plateau-patience", type=int, default=2)
    p.add_argument("--phase1-trough-margin-ratio", type=float, default=1.0)
    p.set_defaults(phase1_prune_enabled=True)
    p.add_argument("--phase1-prune-enabled", dest="phase1_prune_enabled", action="store_true")
    p.add_argument("--phase1-no-prune", dest="phase1_prune_enabled", action="store_false")
    p.add_argument("--phase1-prune-fraction", type=float, default=0.25)
    p.add_argument("--phase1-prune-max-candidates", type=int, default=6)
    p.add_argument("--phase1-prune-max-regression", type=float, default=1e-8)
    p.add_argument(
        "--phase3-motif-source-json",
        type=Path,
        default=None,
        help="Optional solved HH continuation JSON used to derive a transferable motif library for phase3_v1.",
    )
    p.add_argument(
        "--phase3-symmetry-mitigation-mode",
        choices=["off", "verify_only", "postselect_diag_v1", "projector_renorm_v1"],
        default="off",
        help="Optional Phase 3 symmetry hook. verify_only preserves current behavior; active symmetry modes remain metadata/telemetry hooks here and are enforced in the noise oracle path.",
    )
    p.set_defaults(phase3_enable_rescue=False)
    p.add_argument("--phase3-enable-rescue", dest="phase3_enable_rescue", action="store_true")
    p.add_argument("--phase3-no-rescue", dest="phase3_enable_rescue", action="store_false")
    p.add_argument(
        "--phase3-lifetime-cost-mode",
        choices=["off", "phase3_v1"],
        default="phase3_v1",
        help="Enable deterministic lifetime burden weighting inside the existing full_v2 score.",
    )
    p.add_argument(
        "--phase3-runtime-split-mode",
        choices=["off", "shortlist_pauli_children_v1"],
        default="off",
        help="Opt-in shortlist-only macro splitting. When enabled, shortlisted macro generators may be replaced by single-term children at phase2/phase3 scoring time.",
    )
    p.add_argument("--adapt-maxiter", type=int, default=300, help="Inner optimizer maxiter per re-optimization")
    p.add_argument("--adapt-spsa-a", type=float, default=0.2)
    p.add_argument("--adapt-spsa-c", type=float, default=0.1)
    p.add_argument("--adapt-spsa-alpha", type=float, default=0.602)
    p.add_argument("--adapt-spsa-gamma", type=float, default=0.101)
    p.add_argument("--adapt-spsa-A", type=float, default=10.0)
    p.add_argument("--adapt-spsa-avg-last", type=int, default=0)
    p.add_argument("--adapt-spsa-eval-repeats", type=int, default=1)
    p.add_argument(
        "--adapt-spsa-eval-agg",
        choices=["mean", "median"],
        default="mean",
    )
    p.add_argument("--adapt-spsa-callback-every", type=int, default=5)
    p.add_argument("--adapt-spsa-progress-every-s", type=float, default=60.0)
    p.add_argument("--adapt-seed", type=int, default=7)
    p.set_defaults(adapt_allow_repeats=True)
    p.add_argument("--adapt-allow-repeats", dest="adapt_allow_repeats", action="store_true")
    p.add_argument("--adapt-no-repeats", dest="adapt_allow_repeats", action="store_false")
    p.set_defaults(adapt_finite_angle_fallback=True)
    p.add_argument(
        "--adapt-finite-angle-fallback",
        dest="adapt_finite_angle_fallback",
        action="store_true",
        help="If gradients are below threshold, scan finite ±theta probes to continue ADAPT when beneficial.",
    )
    p.add_argument(
        "--adapt-no-finite-angle-fallback",
        dest="adapt_finite_angle_fallback",
        action="store_false",
        help="Disable finite-angle fallback and stop immediately when gradients are below threshold.",
    )
    p.add_argument(
        "--adapt-finite-angle",
        type=float,
        default=0.1,
        help="Probe angle theta used by finite-angle fallback (tests ±theta).",
    )
    p.add_argument(
        "--adapt-finite-angle-min-improvement",
        type=float,
        default=1e-12,
        help="Minimum required energy drop from finite-angle probe to accept fallback selection.",
    )
    p.add_argument(
        "--adapt-disable-hh-seed",
        action="store_true",
        help="Disable HH preconditioning with the compact quadrature seed block.",
    )
    p.add_argument(
        "--adapt-gradient-parity-check",
        action="store_true",
        help=(
            "Debug-only parity guard: compare one reused-Hpsi gradient per ADAPT depth "
            f"against the legacy commutator path (rtol={_ADAPT_GRADIENT_PARITY_RTOL:.1e})."
        ),
    )
    p.add_argument(
        "--adapt-drop-floor",
        type=float,
        default=None,
        help=(
            "Energy-drop floor for plateau stop policy (drop = ΔE_abs(d-1)-ΔE_abs(d)). "
            "If omitted, HH phase1_v1/phase2_v1/phase3_v1 resolves to 5e-4; Hubbard / HH legacy stay off. "
            "Pass a negative value to disable explicitly."
        ),
    )
    p.add_argument(
        "--adapt-drop-patience",
        type=int,
        default=None,
        help=(
            "Consecutive low-drop depth count required to trigger drop plateau stop. "
            "If omitted, HH phase1_v1/phase2_v1/phase3_v1 resolves to 3; Hubbard / HH legacy stay off."
        ),
    )
    p.add_argument(
        "--adapt-drop-min-depth",
        type=int,
        default=None,
        help=(
            "Minimum ADAPT depth before evaluating the drop plateau stop policy. "
            "If omitted, HH phase1_v1/phase2_v1/phase3_v1 resolves to 12; Hubbard / HH legacy stay off."
        ),
    )
    p.add_argument(
        "--adapt-grad-floor",
        type=float,
        default=None,
        help=(
            "Optional secondary gradient floor for drop plateau stop. "
            "If omitted, HH phase1_v1/phase2_v1/phase3_v1 resolves to 2e-2; Hubbard / HH legacy disable it. "
            "Pass a negative value to disable explicitly."
        ),
    )
    p.add_argument(
        "--adapt-eps-energy-min-extra-depth",
        type=int,
        default=-1,
        help=(
            "Minimum extra ADAPT depth before the eps-energy guard can trigger. "
            "Use -1 to auto-set this to L. Telemetry-only in HH phase1_v1/phase2_v1/phase3_v1."
        ),
    )
    p.add_argument(
        "--adapt-eps-energy-patience",
        type=int,
        default=-1,
        help=(
            "Consecutive low-improvement depth count required for the eps-energy guard. "
            "Use -1 to auto-set this to L. Telemetry-only in HH phase1_v1/phase2_v1/phase3_v1."
        ),
    )
    p.add_argument(
        "--adapt-ref-json",
        type=Path,
        default=None,
        help=(
            "Import reference state from an ADAPT/VQE JSON initial_state.amplitudes_qn_to_q0. "
            "In HH phase1_v1/phase2_v1/phase3_v1 reruns, metadata-compatible warm/ADAPT JSON can also "
            "reuse ground_state exact-energy fields."
        ),
    )
    p.add_argument("--paop-r", type=int, default=1, help="Cloud radius R for paop_full/paop_lf_full pools.")
    p.add_argument(
        "--paop-split-paulis",
        action="store_true",
        help="Split composite PAOP generators into single Pauli terms.",
    )
    p.add_argument(
        "--paop-prune-eps",
        type=float,
        default=0.0,
        help="Prune PAOP Pauli terms below this absolute coefficient threshold.",
    )
    p.add_argument(
        "--paop-normalization",
        choices=["none", "fro", "maxcoeff"],
        default="none",
        help="Normalization mode for PAOP generators before ADAPT search.",
    )

    # Trotter dynamics
    p.add_argument("--t-final", type=float, default=20.0)
    p.add_argument("--num-times", type=int, default=201)
    p.add_argument("--suzuki-order", type=int, default=2)
    p.add_argument("--trotter-steps", type=int, default=64)

    p.add_argument("--initial-state-source", choices=["exact", "adapt_vqe", "hf"], default="adapt_vqe")

    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-pdf", type=Path, default=None)
    p.add_argument(
        "--dense-eigh-max-dim",
        type=int,
        default=8192,
        help="Skip full dense Hamiltonian diagonalization when Hilbert dimension exceeds this threshold.",
    )
    p.add_argument("--skip-pdf", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _ai_log("hardcoded_adapt_main_start", settings=vars(args))
    run_command = current_command_string()
    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    json_dir = artifacts_dir / "json"
    pdf_dir = artifacts_dir / "pdf"
    json_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    prob = "hh" if str(args.problem).strip().lower() == "hh" else "hubbard"
    output_json = args.output_json or (json_dir / f"adapt_{prob}_L{args.L}.json")
    output_pdf = args.output_pdf or (pdf_dir / f"adapt_{prob}_L{args.L}.pdf")

    # 1) Build Hamiltonian
    problem_key = str(args.problem).strip().lower()
    if problem_key == "hh":
        h_poly = build_hubbard_holstein_hamiltonian(
            dims=int(args.L),
            J=float(args.t),
            U=float(args.u),
            omega0=float(args.omega0),
            g=float(args.g_ep),
            n_ph_max=int(args.n_ph_max),
            boson_encoding=str(args.boson_encoding),
            v_t=None,
            v0=float(args.dv),
            t_eval=None,
            repr_mode="JW",
            indexing=str(args.ordering),
            pbc=(str(args.boundary) == "periodic"),
            include_zero_point=True,
        )
    else:
        h_poly = build_hubbard_hamiltonian(
            dims=int(args.L),
            t=float(args.t),
            U=float(args.u),
            v=float(args.dv),
            repr_mode="JW",
            indexing=str(args.ordering),
            pbc=(str(args.boundary) == "periodic"),
        )

    native_order, coeff_map_exyz = _collect_hardcoded_terms_exyz(h_poly)
    _ai_log("hardcoded_adapt_hamiltonian_built", L=int(args.L), num_terms=int(len(coeff_map_exyz)))
    if args.term_order == "native":
        ordered_labels_exyz = list(native_order)
    else:
        ordered_labels_exyz = sorted(coeff_map_exyz)
    nq_total = len(ordered_labels_exyz[0]) if ordered_labels_exyz else 2 * int(args.L)
    hilbert_dim = 1 << int(nq_total)
    dense_eigh_enabled = bool(hilbert_dim <= int(args.dense_eigh_max_dim))
    hmat: np.ndarray | None = None
    if dense_eigh_enabled:
        hmat = _build_hamiltonian_matrix(coeff_map_exyz)
    else:
        _ai_log(
            "hardcoded_adapt_dense_eigh_skipped",
            hilbert_dim=int(hilbert_dim),
            dense_eigh_max_dim=int(args.dense_eigh_max_dim),
        )

    psi_ref_override_for_adapt: np.ndarray | None = None
    adapt_ref_import: dict[str, Any] | None = None
    adapt_ref_meta: Mapping[str, Any] | None = None
    adapt_ref_base_depth = 0
    if args.adapt_ref_json is not None:
        psi_ref_override_for_adapt, adapt_ref_meta = _load_adapt_initial_state(
            Path(args.adapt_ref_json),
            int(nq_total),
        )
        adapt_ref_vqe = adapt_ref_meta.get("adapt_vqe", {})
        if isinstance(adapt_ref_vqe, dict):
            ref_depth_raw = adapt_ref_vqe.get("ansatz_depth")
            try:
                ref_depth_val = int(ref_depth_raw)
                if ref_depth_val >= 0:
                    adapt_ref_base_depth = int(ref_depth_val)
            except (TypeError, ValueError):
                adapt_ref_base_depth = 0
        adapt_ref_import = {
            "path": str(Path(args.adapt_ref_json)),
            "initial_state_source": adapt_ref_meta.get("initial_state_source"),
            "settings": adapt_ref_meta.get("settings", {}),
            "adapt_vqe": adapt_ref_meta.get("adapt_vqe", {}),
            "adapt_ref_base_depth": int(adapt_ref_base_depth),
        }
        _ai_log(
            "hardcoded_adapt_ref_json_loaded",
            path=str(Path(args.adapt_ref_json)),
            initial_state_source=adapt_ref_meta.get("initial_state_source"),
            adapt_ref_base_depth=int(adapt_ref_base_depth),
        )

    # Sector-filtered exact ground state: ADAPT-VQE preserves particle number,
    # so compare against the GS within the same (n_alpha, n_beta) sector.
    # For HH: use fermion-only sector filtering (phonon qubits free).
    num_particles_main = half_filled_num_particles(int(args.L))
    gs_energy_exact, gs_energy_source, exact_energy_reuse_mismatches = _resolve_exact_energy_override_from_adapt_ref(
        adapt_ref_meta=adapt_ref_meta,
        args=args,
        problem=problem_key,
        continuation_mode=str(args.adapt_continuation_mode),
    )
    if gs_energy_exact is None:
        gs_energy_exact = _exact_gs_energy_for_problem(
            h_poly,
            problem=problem_key,
            num_sites=int(args.L),
            num_particles=num_particles_main,
            indexing=str(args.ordering),
            n_ph_max=int(args.n_ph_max),
            boson_encoding=str(args.boson_encoding),
            t=float(args.t),
            u=float(args.u),
            dv=float(args.dv),
            omega0=float(args.omega0),
            g_ep=float(args.g_ep),
            boundary=str(args.boundary),
        )
        gs_energy_source = "computed"
    if adapt_ref_import is not None:
        adapt_ref_import["exact_energy_reused"] = bool(gs_energy_source == "adapt_ref_json")
        adapt_ref_import["exact_energy_reuse_mismatches"] = list(exact_energy_reuse_mismatches)
        if gs_energy_source == "adapt_ref_json":
            adapt_ref_import["reused_exact_energy"] = float(gs_energy_exact)
            _ai_log(
                "hardcoded_adapt_exact_energy_reused",
                path=str(Path(args.adapt_ref_json)),
                exact_energy=float(gs_energy_exact),
            )
        elif (
            problem_key == "hh"
            and str(args.adapt_continuation_mode).strip().lower() in _HH_STAGED_CONTINUATION_MODES
        ):
            _ai_log(
                "hardcoded_adapt_exact_energy_reuse_skipped",
                path=str(Path(args.adapt_ref_json)),
                mismatch_count=int(len(exact_energy_reuse_mismatches)),
                has_candidate=bool(_resolve_exact_energy_from_payload(adapt_ref_meta or {})),
            )

    # Full-spectrum eigenvectors are optional (memory heavy for large HH spaces).
    psi_exact_ground: np.ndarray
    if hmat is not None:
        evals_full, evecs_full = np.linalg.eigh(hmat)
        psi_exact_ground_opt: np.ndarray | None = None
        for idx in range(len(evals_full)):
            if abs(evals_full[idx] - gs_energy_exact) < 1e-8:
                psi_exact_ground_opt = _normalize_state(
                    np.asarray(evecs_full[:, idx], dtype=complex).reshape(-1)
                )
                break
        if psi_exact_ground_opt is None:
            gs_idx_fallback = int(np.argmin(evals_full))
            psi_exact_ground_opt = _normalize_state(
                np.asarray(evecs_full[:, gs_idx_fallback], dtype=complex).reshape(-1)
            )
        psi_exact_ground = psi_exact_ground_opt
    elif problem_key == "hh":
        psi_exact_ground = _normalize_state(
            np.asarray(
                hubbard_holstein_reference_state(
                    dims=int(args.L),
                    num_particles=num_particles_main,
                    n_ph_max=int(args.n_ph_max),
                    boson_encoding=str(args.boson_encoding),
                    indexing=str(args.ordering),
                ),
                dtype=complex,
            ).reshape(-1)
        )
    else:
        psi_exact_ground = _normalize_state(
            np.asarray(
                hartree_fock_statevector(int(args.L), num_particles_main, indexing=str(args.ordering)),
                dtype=complex,
            ).reshape(-1)
        )

    # 2) Run ADAPT-VQE
    adapt_payload: dict[str, Any]
    try:
        adapt_payload, psi_adapt = _run_hardcoded_adapt_vqe(
            h_poly=h_poly,
            num_sites=int(args.L),
            ordering=str(args.ordering),
            problem=str(args.problem),
            adapt_pool=(str(args.adapt_pool) if args.adapt_pool is not None else None),
            t=float(args.t),
            u=float(args.u),
            dv=float(args.dv),
            boundary=str(args.boundary),
            omega0=float(args.omega0),
            g_ep=float(args.g_ep),
            n_ph_max=int(args.n_ph_max),
            boson_encoding=str(args.boson_encoding),
            max_depth=int(args.adapt_max_depth),
            eps_grad=float(args.adapt_eps_grad),
            eps_energy=float(args.adapt_eps_energy),
            maxiter=int(args.adapt_maxiter),
            seed=int(args.adapt_seed),
            adapt_inner_optimizer=str(args.adapt_inner_optimizer),
            adapt_spsa_a=float(args.adapt_spsa_a),
            adapt_spsa_c=float(args.adapt_spsa_c),
            adapt_spsa_alpha=float(args.adapt_spsa_alpha),
            adapt_spsa_gamma=float(args.adapt_spsa_gamma),
            adapt_spsa_A=float(args.adapt_spsa_A),
            adapt_spsa_avg_last=int(args.adapt_spsa_avg_last),
            adapt_spsa_eval_repeats=int(args.adapt_spsa_eval_repeats),
            adapt_spsa_eval_agg=str(args.adapt_spsa_eval_agg),
            adapt_spsa_callback_every=int(args.adapt_spsa_callback_every),
            adapt_spsa_progress_every_s=float(args.adapt_spsa_progress_every_s),
            adapt_state_backend=str(args.adapt_state_backend),
            adapt_reopt_policy=str(args.adapt_reopt_policy),
            adapt_window_size=int(args.adapt_window_size),
            adapt_window_topk=int(args.adapt_window_topk),
            adapt_full_refit_every=int(args.adapt_full_refit_every),
            adapt_final_full_refit=bool(str(args.adapt_final_full_refit).strip().lower() == "true"),
            adapt_continuation_mode=str(args.adapt_continuation_mode),
            allow_repeats=bool(args.adapt_allow_repeats),
            finite_angle_fallback=bool(args.adapt_finite_angle_fallback),
            finite_angle=float(args.adapt_finite_angle),
            finite_angle_min_improvement=float(args.adapt_finite_angle_min_improvement),
            adapt_drop_floor=(float(args.adapt_drop_floor) if args.adapt_drop_floor is not None else None),
            adapt_drop_patience=(int(args.adapt_drop_patience) if args.adapt_drop_patience is not None else None),
            adapt_drop_min_depth=(int(args.adapt_drop_min_depth) if args.adapt_drop_min_depth is not None else None),
            adapt_grad_floor=(float(args.adapt_grad_floor) if args.adapt_grad_floor is not None else None),
            adapt_eps_energy_min_extra_depth=int(args.adapt_eps_energy_min_extra_depth),
            adapt_eps_energy_patience=int(args.adapt_eps_energy_patience),
            adapt_ref_base_depth=int(adapt_ref_base_depth),
            paop_r=int(args.paop_r),
            paop_split_paulis=bool(args.paop_split_paulis),
            paop_prune_eps=float(args.paop_prune_eps),
            paop_normalization=str(args.paop_normalization),
            disable_hh_seed=bool(args.adapt_disable_hh_seed),
            psi_ref_override=psi_ref_override_for_adapt,
            adapt_gradient_parity_check=bool(args.adapt_gradient_parity_check),
            exact_gs_override=float(gs_energy_exact),
            phase1_lambda_F=float(args.phase1_lambda_F),
            phase1_lambda_compile=float(args.phase1_lambda_compile),
            phase1_lambda_measure=float(args.phase1_lambda_measure),
            phase1_lambda_leak=float(args.phase1_lambda_leak),
            phase1_score_z_alpha=float(args.phase1_score_z_alpha),
            phase1_probe_max_positions=int(args.phase1_probe_max_positions),
            phase1_plateau_patience=int(args.phase1_plateau_patience),
            phase1_trough_margin_ratio=float(args.phase1_trough_margin_ratio),
            phase1_prune_enabled=bool(args.phase1_prune_enabled),
            phase1_prune_fraction=float(args.phase1_prune_fraction),
            phase1_prune_max_candidates=int(args.phase1_prune_max_candidates),
            phase1_prune_max_regression=float(args.phase1_prune_max_regression),
            phase3_motif_source_json=(Path(args.phase3_motif_source_json) if args.phase3_motif_source_json is not None else None),
            phase3_symmetry_mitigation_mode=str(args.phase3_symmetry_mitigation_mode),
            phase3_enable_rescue=bool(args.phase3_enable_rescue),
            phase3_lifetime_cost_mode=str(args.phase3_lifetime_cost_mode),
            phase3_runtime_split_mode=str(args.phase3_runtime_split_mode),
        )
    except Exception as exc:
        _ai_log("hardcoded_adapt_vqe_failed", L=int(args.L), error=str(exc))
        adapt_payload = {
            "success": False,
            "method": f"hardcoded_adapt_vqe_{str(args.adapt_pool).lower()}",
            "energy": None,
            "adapt_inner_optimizer": str(args.adapt_inner_optimizer),
            "error": str(exc),
        }
        if str(args.adapt_inner_optimizer).strip().upper() == "SPSA":
            adapt_payload["adapt_spsa"] = {
                "a": float(args.adapt_spsa_a),
                "c": float(args.adapt_spsa_c),
                "alpha": float(args.adapt_spsa_alpha),
                "gamma": float(args.adapt_spsa_gamma),
                "A": float(args.adapt_spsa_A),
                "avg_last": int(args.adapt_spsa_avg_last),
                "eval_repeats": int(args.adapt_spsa_eval_repeats),
                "eval_agg": str(args.adapt_spsa_eval_agg),
            }
        psi_adapt = psi_exact_ground

    # 3) Select initial state for dynamics
    num_particles = half_filled_num_particles(int(args.L))
    if problem_key == "hh":
        psi_hf = _normalize_state(
            np.asarray(
                hubbard_holstein_reference_state(
                    dims=int(args.L),
                    num_particles=num_particles,
                    n_ph_max=int(args.n_ph_max),
                    boson_encoding=str(args.boson_encoding),
                    indexing=str(args.ordering),
                ),
                dtype=complex,
            ).reshape(-1)
        )
    else:
        psi_hf = _normalize_state(
            np.asarray(
                hartree_fock_statevector(int(args.L), num_particles, indexing=str(args.ordering)),
                dtype=complex,
            ).reshape(-1)
        )

    if args.initial_state_source == "adapt_vqe" and bool(adapt_payload.get("success", False)):
        psi0 = psi_adapt
        _ai_log("hardcoded_adapt_initial_state_selected", source="adapt_vqe")
    elif args.initial_state_source == "adapt_vqe":
        raise RuntimeError("Requested --initial-state-source adapt_vqe but ADAPT-VQE failed.")
    elif args.initial_state_source == "hf":
        psi0 = psi_hf
        _ai_log("hardcoded_adapt_initial_state_selected", source="hf")
    else:
        psi0 = psi_exact_ground
        _ai_log("hardcoded_adapt_initial_state_selected", source="exact")

    # 4) Trajectory
    if hmat is None:
        trajectory = []
        _ai_log(
            "hardcoded_adapt_trajectory_skipped_no_dense_hmat",
            hilbert_dim=int(hilbert_dim),
            dense_eigh_max_dim=int(args.dense_eigh_max_dim),
        )
    else:
        trajectory, _exact_states = _simulate_trajectory(
            num_sites=int(args.L),
            psi0=psi0,
            hmat=hmat,
            ordered_labels_exyz=ordered_labels_exyz,
            coeff_map_exyz=coeff_map_exyz,
            trotter_steps=int(args.trotter_steps),
            t_final=float(args.t_final),
            num_times=int(args.num_times),
            suzuki_order=int(args.suzuki_order),
        )

    # 5) Emit JSON
    payload: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "hardcoded_adapt",
        "settings": {
            "L": int(args.L),
            "t": float(args.t),
            "u": float(args.u),
            "problem": str(args.problem),
            "omega0": float(args.omega0),
            "g_ep": float(args.g_ep),
            "n_ph_max": int(args.n_ph_max),
            "boson_encoding": str(args.boson_encoding),
            "dv": float(args.dv),
            "boundary": str(args.boundary),
            "ordering": str(args.ordering),
            "t_final": float(args.t_final),
            "num_times": int(args.num_times),
            "suzuki_order": int(args.suzuki_order),
            "trotter_steps": int(args.trotter_steps),
            "term_order": str(args.term_order),
            "dense_eigh_max_dim": int(args.dense_eigh_max_dim),
            "dense_eigh_enabled": bool(dense_eigh_enabled),
            "hilbert_dim": int(hilbert_dim),
            "adapt_pool": (str(args.adapt_pool) if args.adapt_pool is not None else None),
            "adapt_continuation_mode": str(args.adapt_continuation_mode),
            "adapt_max_depth": int(args.adapt_max_depth),
            "adapt_eps_grad": float(args.adapt_eps_grad),
            "adapt_eps_energy": float(args.adapt_eps_energy),
            "adapt_inner_optimizer": str(args.adapt_inner_optimizer),
            "adapt_state_backend": str(args.adapt_state_backend),
            "adapt_finite_angle_fallback": bool(args.adapt_finite_angle_fallback),
            "adapt_finite_angle": float(args.adapt_finite_angle),
            "adapt_finite_angle_min_improvement": float(args.adapt_finite_angle_min_improvement),
            "adapt_drop_floor": (float(args.adapt_drop_floor) if args.adapt_drop_floor is not None else None),
            "adapt_drop_patience": (int(args.adapt_drop_patience) if args.adapt_drop_patience is not None else None),
            "adapt_drop_min_depth": (int(args.adapt_drop_min_depth) if args.adapt_drop_min_depth is not None else None),
            "adapt_grad_floor": (float(args.adapt_grad_floor) if args.adapt_grad_floor is not None else None),
            "adapt_drop_floor_resolved": adapt_payload.get("adapt_drop_floor_resolved"),
            "adapt_drop_patience_resolved": adapt_payload.get("adapt_drop_patience_resolved"),
            "adapt_drop_min_depth_resolved": adapt_payload.get("adapt_drop_min_depth_resolved"),
            "adapt_grad_floor_resolved": adapt_payload.get("adapt_grad_floor_resolved"),
            "adapt_drop_floor_source": adapt_payload.get("adapt_drop_floor_source"),
            "adapt_drop_patience_source": adapt_payload.get("adapt_drop_patience_source"),
            "adapt_drop_min_depth_source": adapt_payload.get("adapt_drop_min_depth_source"),
            "adapt_grad_floor_source": adapt_payload.get("adapt_grad_floor_source"),
            "adapt_drop_policy_source": adapt_payload.get("adapt_drop_policy_source"),
            "adapt_eps_energy_min_extra_depth": int(args.adapt_eps_energy_min_extra_depth),
            "adapt_eps_energy_patience": int(args.adapt_eps_energy_patience),
            "adapt_ref_base_depth": int(adapt_ref_base_depth),
            "adapt_gradient_parity_check": bool(args.adapt_gradient_parity_check),
            "adapt_seed": int(args.adapt_seed),
            "adapt_reopt_policy": str(args.adapt_reopt_policy),
            "adapt_window_size": int(args.adapt_window_size),
            "adapt_window_topk": int(args.adapt_window_topk),
            "adapt_full_refit_every": int(args.adapt_full_refit_every),
            "adapt_final_full_refit": str(args.adapt_final_full_refit),
            "phase1_lambda_F": float(args.phase1_lambda_F),
            "phase1_lambda_compile": float(args.phase1_lambda_compile),
            "phase1_lambda_measure": float(args.phase1_lambda_measure),
            "phase1_lambda_leak": float(args.phase1_lambda_leak),
            "phase1_score_z_alpha": float(args.phase1_score_z_alpha),
            "phase1_probe_max_positions": int(args.phase1_probe_max_positions),
            "phase1_plateau_patience": int(args.phase1_plateau_patience),
            "phase1_trough_margin_ratio": float(args.phase1_trough_margin_ratio),
            "phase1_prune_enabled": bool(args.phase1_prune_enabled),
            "phase1_prune_fraction": float(args.phase1_prune_fraction),
            "phase1_prune_max_candidates": int(args.phase1_prune_max_candidates),
            "phase1_prune_max_regression": float(args.phase1_prune_max_regression),
            "phase3_motif_source_json": (
                str(args.phase3_motif_source_json)
                if args.phase3_motif_source_json is not None
                else None
            ),
            "phase3_symmetry_mitigation_mode": str(args.phase3_symmetry_mitigation_mode),
            "phase3_enable_rescue": bool(args.phase3_enable_rescue),
            "phase3_lifetime_cost_mode": str(args.phase3_lifetime_cost_mode),
            "phase3_runtime_split_mode": str(args.phase3_runtime_split_mode),
            "adapt_ref_json": (str(args.adapt_ref_json) if args.adapt_ref_json is not None else None),
            "paop_r": int(args.paop_r),
            "paop_split_paulis": bool(args.paop_split_paulis),
            "paop_prune_eps": float(args.paop_prune_eps),
            "paop_normalization": str(args.paop_normalization),
            "initial_state_source": str(args.initial_state_source),
        },
        "hamiltonian": {
            "num_qubits": int(
                len(ordered_labels_exyz[0])
                if ordered_labels_exyz
                else int(round(math.log2(hmat.shape[0])))
            ),
            "num_terms": int(len(coeff_map_exyz)),
            "coefficients_exyz": [
                {
                    "label_exyz": lbl,
                    "coeff": {"re": float(np.real(coeff_map_exyz[lbl])), "im": float(np.imag(coeff_map_exyz[lbl]))},
                }
                for lbl in ordered_labels_exyz
            ],
        },
        "ground_state": {
            "exact_energy": float(gs_energy_exact),
            "exact_energy_source": str(gs_energy_source),
            "method": (EXACT_METHOD if hmat is not None else "sector_exact_only_no_dense_eigh"),
        },
        "adapt_vqe": adapt_payload,
        "initial_state": {
            "source": str(args.initial_state_source if args.initial_state_source != "adapt_vqe" or adapt_payload.get("success") else "exact"),
            "amplitudes_qn_to_q0": _state_to_amplitudes_qn_to_q0(psi0),
            "handoff_state_kind": (
                "prepared_state"
                if (args.initial_state_source == "adapt_vqe" and bool(adapt_payload.get("success", False)))
                else "reference_state"
            ),
        },
        "trajectory": trajectory,
    }
    if str(args.adapt_inner_optimizer).strip().upper() == "SPSA":
        payload["settings"]["adapt_spsa"] = {
            "a": float(args.adapt_spsa_a),
            "c": float(args.adapt_spsa_c),
            "alpha": float(args.adapt_spsa_alpha),
            "gamma": float(args.adapt_spsa_gamma),
            "A": float(args.adapt_spsa_A),
            "avg_last": int(args.adapt_spsa_avg_last),
            "eval_repeats": int(args.adapt_spsa_eval_repeats),
            "eval_agg": str(args.adapt_spsa_eval_agg),
            "callback_every": int(args.adapt_spsa_callback_every),
            "progress_every_s": float(args.adapt_spsa_progress_every_s),
        }
    if adapt_ref_import is not None:
        payload["adapt_ref_import"] = adapt_ref_import

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not args.skip_pdf:
        _write_pipeline_pdf(output_pdf, payload, run_command)

    _ai_log(
        "hardcoded_adapt_main_done",
        L=int(args.L),
        output_json=str(output_json),
        output_pdf=(str(output_pdf) if not args.skip_pdf else None),
        adapt_energy=adapt_payload.get("energy"),
    )
    print(f"Wrote JSON: {output_json}")
    if not args.skip_pdf:
        print(f"Wrote PDF:  {output_pdf}")


if __name__ == "__main__":
    main()
