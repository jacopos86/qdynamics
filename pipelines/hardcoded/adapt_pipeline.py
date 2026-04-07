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
import copy
import errno
import hashlib
import json
import math
import os
import re
import sys
import time
import weakref
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
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
    render_command_page,
    render_text_page,
    current_command_string,
)
from docs.reports.report_pages import (
    render_executive_summary_page,
    render_manifest_overview_page,
    render_section_divider_page,
)

# Module-level aliases used by the plotting body
plt = get_plt() if HAS_MATPLOTLIB else None  # type: ignore[assignment]
PdfPages = get_PdfPages() if HAS_MATPLOTLIB else type("PdfPages", (), {})  # type: ignore[misc]

# ---------------------------------------------------------------------------
# Imports from the active repo quantum modules (no pydephasing fallback).
# ---------------------------------------------------------------------------
from src.quantum.hubbard_latex_python_pairs import (
    bravais_nearest_neighbor_edges,
    build_hubbard_hamiltonian,
    build_hubbard_holstein_hamiltonian,
    boson_qubits_per_site,
)
from src.quantum.hartree_fock_reference_state import hartree_fock_statevector
from src.quantum.ansatz_parameterization import (
    AnsatzParameterLayout,
    build_parameter_layout,
    project_runtime_theta_block_mean,
    runtime_indices_for_logical_indices,
    runtime_insert_position,
    serialize_layout,
)
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
    apply_exp_pauli_polynomial_termwise,
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
    MaturePruneTrial,
    Phase2OptimizerMemoryAdapter,
    PhaseControllerSnapshot,
    ScaffoldCoordinateMetadata,
    ScaffoldFingerprintLite,
)
from pipelines.hardcoded.hh_continuation_generators import (
    build_pool_generator_registry,
    build_runtime_split_child_sets,
    build_runtime_split_children,
    build_split_event,
    selected_generator_metadata_for_labels,
)
from pipelines.hardcoded.handoff_state_bundle import build_statevector_manifest
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
    family_repeat_cost_from_history,
    greedy_batch_select,
    measurement_group_keys_for_term,
    phase_shortlist_records,
    raw_f_metric_from_state,
    reduced_plane_batch_select,
    shortlist_records,
)
from pipelines.hardcoded.hh_continuation_pruning import (
    PruneConfig,
    apply_pruning,
    cheap_prune_score,
    post_prune_refit,
    rank_prune_candidates,
)
from pipelines.hardcoded.hh_backend_compile_oracle import (
    BackendCompileConfig,
    BackendCompileOracle,
)

try:
    from src.quantum.operator_pools import make_pool as make_paop_pool
except Exception as exc:  # pragma: no cover - defensive fallback
    make_paop_pool = None
    _PAOP_IMPORT_ERROR = str(exc)
else:
    _PAOP_IMPORT_ERROR = ""

try:
    from src.quantum.operator_pools.polaron_paop import make_phonon_motifs
except Exception:  # pragma: no cover - optional legacy product-family seam
    make_phonon_motifs = None

try:
    from src.quantum.operator_pools.vlf_sq import build_vlf_sq_pool as build_vlf_sq_family
except Exception:  # pragma: no cover - optional VLF/SQ family seam
    build_vlf_sq_family = None

EXACT_LABEL = "Exact_Hardcode"
EXACT_METHOD = "python_matrix_eigendecomposition"
_ADAPT_GRADIENT_PARITY_RTOL = 1e-8
_STDOUT_PIPE_BROKEN = False

PAULI_MATS = {
    "e": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
    "x": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


def _safe_stdout_print(*args: Any, **kwargs: Any) -> bool:
    global _STDOUT_PIPE_BROKEN
    if _STDOUT_PIPE_BROKEN:
        return False
    try:
        print(*args, **kwargs)
        return True
    except BrokenPipeError:
        _STDOUT_PIPE_BROKEN = True
        return False
    except OSError as exc:
        if getattr(exc, "errno", None) == errno.EPIPE:
            _STDOUT_PIPE_BROKEN = True
            return False
        raise


def _ai_log(event: str, **fields: Any) -> None:
    payload = {
        "event": str(event),
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    _safe_stdout_print(f"AI_LOG {json.dumps(payload, sort_keys=True, default=str)}", flush=True)


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
    stored_nq_total_raw = initial_state.get("nq_total", None)
    if stored_nq_total_raw is not None and int(stored_nq_total_raw) != int(nq_total):
        raise ValueError(
            f"ADAPT input JSON initial_state.nq_total={int(stored_nq_total_raw)} does not match expected nq_total={int(nq_total)}."
        )
    amplitudes = initial_state.get("amplitudes_qn_to_q0")
    psi = _state_from_amplitudes_qn_to_q0(amplitudes, int(nq_total))
    meta = {
        "settings": raw.get("settings", {}),
        "adapt_vqe": raw.get("adapt_vqe", {}),
        "ground_state": raw.get("ground_state", {}),
        "vqe": raw.get("vqe", {}),
        "initial_state_source": initial_state.get("source"),
        "initial_state_handoff_state_kind": initial_state.get("handoff_state_kind"),
    }
    return psi, meta


def _default_adapt_input_state(
    *,
    problem: str,
    num_sites: int,
    ordering: str,
    n_ph_max: int,
    boson_encoding: str,
) -> tuple[np.ndarray, str, str]:
    problem_key = str(problem).strip().lower()
    num_particles = half_filled_num_particles(int(num_sites))
    if problem_key == "hh":
        psi = _normalize_state(
            np.asarray(
                hubbard_holstein_reference_state(
                    dims=int(num_sites),
                    num_particles=num_particles,
                    n_ph_max=int(n_ph_max),
                    boson_encoding=str(boson_encoding),
                    indexing=str(ordering),
                ),
                dtype=complex,
            ).reshape(-1)
        )
    else:
        psi = _normalize_state(
            np.asarray(
                hartree_fock_statevector(
                    int(num_sites),
                    num_particles,
                    indexing=str(ordering),
                ),
                dtype=complex,
            ).reshape(-1)
        )
    return psi, "hf", "reference_state"


_HH_STAGED_CONTINUATION_MODES = frozenset({"phase1_v1", "phase2_v1", "phase3_v1"})
_HH_UCCSD_PAOP_PRODUCT_SPECS: dict[str, dict[str, Any]] = {
    "uccsd_otimes_paop_lf_std": {
        "motif_family": "paop_lf_std",
        "parameterization": "single_product",
        "adapt_visible": True,
    },
    "uccsd_otimes_paop_lf2_std": {
        "motif_family": "paop_lf2_std",
        "parameterization": "single_product",
        "adapt_visible": True,
    },
    "uccsd_otimes_paop_bond_disp_std": {
        "motif_family": "paop_bond_disp_std",
        "parameterization": "single_product",
        "adapt_visible": True,
    },
    "uccsd_otimes_paop_lf_std_seq2p": {
        "motif_family": "paop_lf_std",
        "parameterization": "double_sequential",
        "adapt_visible": True,
    },
    "uccsd_otimes_paop_lf2_std_seq2p": {
        "motif_family": "paop_lf2_std",
        "parameterization": "double_sequential",
        "adapt_visible": True,
    },
    "uccsd_otimes_paop_bond_disp_std_seq2p": {
        "motif_family": "paop_bond_disp_std",
        "parameterization": "double_sequential",
        "adapt_visible": True,
    },
}
_UCCSD_SINGLE_LABEL_RE = re.compile(r"^uccsd_sing\((alpha|beta):(\d+)->(\d+)\)$")
_UCCSD_DOUBLE_LABEL_RE = re.compile(r"^uccsd_dbl\((aa|bb|ab):(\d+),(\d+)->(\d+),(\d+)\)$")


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


@dataclass(frozen=True)
class _ADAPTLogicalCandidate:
    """Private ADAPT selection unit for multi-parameter logical pool elements."""
    logical_label: str
    pool_indices: tuple[int, ...]
    parameterization: str
    family_id: str


@dataclass
class _BeamBranchState:
    branch_id: int
    parent_branch_id: int | None
    depth_local: int
    terminated: bool
    stop_reason: str | None
    selected_ops: list[AnsatzTerm]
    theta: np.ndarray
    energy_current: float
    available_indices: set[int]
    selection_counts: np.ndarray
    history: list[dict[str, Any]]
    phase1_stage: StageController
    phase1_residual_opened: bool
    phase1_last_probe_reason: str
    phase1_last_positions_considered: list[int]
    phase1_last_trough_detected: bool
    phase1_last_trough_probe_triggered: bool
    phase1_last_selected_score: float | None
    phase1_features_history: list[dict[str, Any]]
    phase1_stage_events: list[dict[str, Any]]
    phase1_measure_cache: MeasurementCacheAudit
    phase1_last_retained_records: list[dict[str, Any]]
    phase2_optimizer_memory: dict[str, Any]
    phase2_last_shortlist_records: list[dict[str, Any]]
    phase2_last_geometric_shortlist_records: list[dict[str, Any]]
    phase2_last_retained_shortlist_records: list[dict[str, Any]]
    phase2_last_admitted_records: list[dict[str, Any]]
    phase2_last_batch_selected: bool
    phase2_last_batch_penalty_total: float
    phase2_last_optimizer_memory_reused: bool
    phase2_last_optimizer_memory_source: str
    phase2_last_shortlist_eval_records: list[dict[str, Any]]
    drop_prev_delta_abs: float
    drop_plateau_hits: int
    eps_energy_low_streak: int
    phase3_split_events: list[dict[str, Any]]
    phase3_runtime_split_summary: dict[str, Any]
    phase3_motif_usage: dict[str, Any]
    phase3_rescue_history: list[dict[str, Any]]
    phase1_prune_metadata: list[ScaffoldCoordinateMetadata]
    phase1_prune_first_seen_steps: dict[str, int]
    phase1_last_prune_summary: dict[str, Any]
    last_transition_kind: str
    last_admission_record_count: int
    cumulative_selector_score: float
    cumulative_selector_burden: float
    nfev_total_local: int

    def clone_for_child(self, *, branch_id: int) -> "_BeamBranchState":
        return _BeamBranchState(
            branch_id=int(branch_id),
            parent_branch_id=int(self.branch_id),
            depth_local=int(self.depth_local),
            terminated=bool(self.terminated),
            stop_reason=(None if self.stop_reason is None else str(self.stop_reason)),
            selected_ops=list(self.selected_ops),
            theta=np.asarray(self.theta, dtype=float).copy(),
            energy_current=float(self.energy_current),
            available_indices=set(int(x) for x in self.available_indices),
            selection_counts=np.asarray(self.selection_counts, dtype=np.int64).copy(),
            history=copy.deepcopy(self.history),
            phase1_stage=self.phase1_stage.clone(),
            phase1_residual_opened=bool(self.phase1_residual_opened),
            phase1_last_probe_reason=str(self.phase1_last_probe_reason),
            phase1_last_positions_considered=[int(x) for x in self.phase1_last_positions_considered],
            phase1_last_trough_detected=bool(self.phase1_last_trough_detected),
            phase1_last_trough_probe_triggered=bool(self.phase1_last_trough_probe_triggered),
            phase1_last_selected_score=(
                None if self.phase1_last_selected_score is None else float(self.phase1_last_selected_score)
            ),
            phase1_features_history=copy.deepcopy(self.phase1_features_history),
            phase1_stage_events=copy.deepcopy(self.phase1_stage_events),
            phase1_measure_cache=self.phase1_measure_cache.clone(),
            phase1_last_retained_records=copy.deepcopy(self.phase1_last_retained_records),
            phase2_optimizer_memory=copy.deepcopy(self.phase2_optimizer_memory),
            phase2_last_shortlist_records=copy.deepcopy(self.phase2_last_shortlist_records),
            phase2_last_geometric_shortlist_records=copy.deepcopy(self.phase2_last_geometric_shortlist_records),
            phase2_last_retained_shortlist_records=copy.deepcopy(self.phase2_last_retained_shortlist_records),
            phase2_last_admitted_records=copy.deepcopy(self.phase2_last_admitted_records),
            phase2_last_batch_selected=bool(self.phase2_last_batch_selected),
            phase2_last_batch_penalty_total=float(self.phase2_last_batch_penalty_total),
            phase2_last_optimizer_memory_reused=bool(self.phase2_last_optimizer_memory_reused),
            phase2_last_optimizer_memory_source=str(self.phase2_last_optimizer_memory_source),
            phase2_last_shortlist_eval_records=copy.deepcopy(self.phase2_last_shortlist_eval_records),
            drop_prev_delta_abs=float(self.drop_prev_delta_abs),
            drop_plateau_hits=int(self.drop_plateau_hits),
            eps_energy_low_streak=int(self.eps_energy_low_streak),
            phase3_split_events=copy.deepcopy(self.phase3_split_events),
            phase3_runtime_split_summary=copy.deepcopy(self.phase3_runtime_split_summary),
            phase3_motif_usage=copy.deepcopy(self.phase3_motif_usage),
            phase3_rescue_history=copy.deepcopy(self.phase3_rescue_history),
            phase1_prune_metadata=[
                ScaffoldCoordinateMetadata(**dict(x.__dict__)) for x in self.phase1_prune_metadata
            ],
            phase1_prune_first_seen_steps={
                str(k): int(v) for k, v in self.phase1_prune_first_seen_steps.items()
            },
            phase1_last_prune_summary=copy.deepcopy(self.phase1_last_prune_summary),
            last_transition_kind=str(self.last_transition_kind),
            last_admission_record_count=int(self.last_admission_record_count),
            cumulative_selector_score=float(self.cumulative_selector_score),
            cumulative_selector_burden=float(self.cumulative_selector_burden),
            nfev_total_local=int(self.nfev_total_local),
        )


@dataclass(frozen=True)
class _BranchExpansionPlan:
    candidate_pool_index: int
    position_id: int
    selection_mode: str
    candidate_label: str
    candidate_term: AnsatzTerm
    feature_row: dict[str, Any] | None
    init_theta: float = 0.0


@dataclass
class _BranchStepScratch:
    energy_current: float
    psi_current: np.ndarray
    hpsi_current: np.ndarray
    gradients: np.ndarray
    grad_magnitudes: np.ndarray
    max_grad: float
    gradient_eval_elapsed_s: float
    append_position: int
    best_idx: int
    selected_position: int
    selection_mode: str
    stage_name: str
    phase1_feature_selected: dict[str, Any] | None
    phase1_stage_transition_reason: str
    phase1_stage_now: str
    phase1_stage_after_transition: StageController
    phase1_last_probe_reason: str
    phase1_last_positions_considered: list[int]
    phase1_last_trough_detected: bool
    phase1_last_trough_probe_triggered: bool
    phase1_last_selected_score: float | None
    phase1_last_retained_records: list[dict[str, Any]]
    phase2_last_shortlist_records: list[dict[str, Any]]
    phase2_last_geometric_shortlist_records: list[dict[str, Any]]
    phase2_last_retained_shortlist_records: list[dict[str, Any]]
    phase2_last_admitted_records: list[dict[str, Any]]
    phase2_last_batch_selected: bool
    phase2_last_batch_penalty_total: float
    phase2_last_optimizer_memory_reused: bool
    phase2_last_optimizer_memory_source: str
    phase2_last_shortlist_eval_records: list[dict[str, Any]]
    phase1_residual_opened: bool
    available_indices_after_transition: set[int]
    phase1_stage_events_after_transition: list[dict[str, Any]]
    phase3_runtime_split_summary_after_eval: dict[str, Any]
    proposals: list[_BranchExpansionPlan]
    stop_reason: str | None
    fallback_scan_size: int
    fallback_best_probe_delta_e: float | None
    fallback_best_probe_theta: float | None


def _parse_seq2p_step_label(label: str) -> tuple[str, str] | None:
    raw = str(label).strip()
    for step_name in ("ferm", "motif"):
        suffix = f"::step={step_name}"
        if raw.endswith(suffix):
            return raw[: -len(suffix)], step_name
    return None


def _build_seq2p_logical_candidates(
    pool: Sequence[AnsatzTerm],
    *,
    family_id: str,
) -> list[_ADAPTLogicalCandidate]:
    candidates: list[_ADAPTLogicalCandidate] = []
    idx = 0
    while idx < len(pool):
        if idx + 1 >= len(pool):
            raise ValueError("Malformed seq2p pool: trailing unpaired flat term.")
        lhs = _parse_seq2p_step_label(str(pool[idx].label))
        rhs = _parse_seq2p_step_label(str(pool[idx + 1].label))
        if lhs is None or rhs is None:
            raise ValueError("Malformed seq2p pool: expected paired ::step=ferm/::step=motif labels.")
        lhs_base, lhs_step = lhs
        rhs_base, rhs_step = rhs
        if lhs_step != "ferm" or rhs_step != "motif" or lhs_base != rhs_base:
            raise ValueError("Malformed seq2p pool: expected adjacent ferm/motif terms for each logical pair.")
        candidates.append(
            _ADAPTLogicalCandidate(
                logical_label=str(lhs_base),
                pool_indices=(int(idx), int(idx + 1)),
                parameterization="double_sequential",
                family_id=str(family_id),
            )
        )
        idx += 2
    return candidates


def _logical_candidate_gradient_summary(
    candidate: _ADAPTLogicalCandidate,
    gradients: np.ndarray,
) -> tuple[float, list[float], list[float]]:
    signed_components = [float(gradients[int(idx)]) for idx in candidate.pool_indices]
    abs_components = [abs(float(val)) for val in signed_components]
    score = math.sqrt(sum(float(val) * float(val) for val in abs_components))
    return float(score), signed_components, abs_components


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

    # 2) Augment with the same lifted HH UCCSD macro generators used by the
    # warm-start path. Do not expose individual Pauli fragments here: they can
    # break the fermion sector even when the parent macro generator preserves it.
    n_sites = int(num_sites)
    pool.extend(
        _build_hh_uccsd_fermion_lifted_pool(
            int(num_sites),
            int(n_ph_max),
            str(boson_encoding),
            str(ordering),
            str(boundary),
            num_particles=tuple(half_filled_num_particles(n_sites)),
        )
    )

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


def _build_vlf_sq_pool(
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
) -> tuple[list[AnsatzTerm], dict[str, Any]]:
    if build_vlf_sq_family is None:
        raise RuntimeError(f"VLF/SQ pool requested but operator_pools module unavailable: {_PAOP_IMPORT_ERROR}")
    if bool(paop_split_paulis):
        raise ValueError("VLF/SQ macro families do not support --paop-split-paulis; keep grouped macro generators intact.")
    pool_specs, meta = build_vlf_sq_family(
        pool_key,
        num_sites=int(num_sites),
        num_particles=tuple(num_particles),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        ordering=str(ordering),
        boundary=str(boundary),
        shell_radius=None,
        prune_eps=float(paop_prune_eps),
        normalization=str(paop_normalization),
    )
    return [AnsatzTerm(label=label, polynomial=poly) for label, poly in pool_specs], dict(meta)


def _clean_real_pool_polynomial(poly: Any, prune_eps: float = 0.0) -> PauliPolynomial:
    terms = poly.return_polynomial()
    if not terms:
        return PauliPolynomial("JW")
    nq = int(terms[0].nqubit())
    cleaned = PauliPolynomial("JW")
    for term in terms:
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(prune_eps):
            continue
        if abs(coeff.imag) > 1e-10:
            raise ValueError(f"Non-negligible imaginary coefficient in product-family pool term: {coeff}")
        cleaned.add_term(PauliTerm(nq, ps=str(term.pw2strng()), pc=float(coeff.real)))
    cleaned._reduce()
    return cleaned


def _fermion_mode_to_site(mode: int, *, num_sites: int, ordering: str) -> int:
    mode_i = int(mode)
    n_sites = int(num_sites)
    ordering_key = str(ordering).strip().lower()
    if mode_i < 0 or mode_i >= 2 * n_sites:
        raise ValueError(f"Fermion mode {mode_i} out of range for num_sites={n_sites}")
    if ordering_key == "interleaved":
        return mode_i // 2
    if ordering_key == "blocked":
        if mode_i < n_sites:
            return mode_i
        return mode_i - n_sites
    raise ValueError(f"Unsupported fermion ordering '{ordering}'.")


def _parse_lifted_uccsd_support(
    label: str,
    *,
    num_sites: int,
    ordering: str,
) -> tuple[str, tuple[int, ...]]:
    raw = str(label).strip()
    prefix = "uccsd_ferm_lifted::"
    if not raw.startswith(prefix):
        raise ValueError(f"Unsupported lifted UCCSD label '{raw}'.")
    body = raw[len(prefix):]

    m_single = _UCCSD_SINGLE_LABEL_RE.match(body)
    if m_single is not None:
        modes = [int(m_single.group(2)), int(m_single.group(3))]
        kind = "single"
    else:
        m_double = _UCCSD_DOUBLE_LABEL_RE.match(body)
        if m_double is None:
            raise ValueError(f"Could not parse lifted UCCSD label '{raw}'.")
        modes = [
            int(m_double.group(2)),
            int(m_double.group(3)),
            int(m_double.group(4)),
            int(m_double.group(5)),
        ]
        kind = "double"

    sites = tuple(
        sorted(
            {
                _fermion_mode_to_site(mode, num_sites=int(num_sites), ordering=str(ordering))
                for mode in modes
            }
        )
    )
    return kind, sites


def _motif_matches_excitation_support(
    *,
    motif: Any,
    motif_family: str,
    support_sites: tuple[int, ...],
    nearest_neighbor_bonds: set[tuple[int, int]],
) -> bool:
    support_set = {int(site) for site in support_sites}
    motif_sites = {int(site) for site in getattr(motif, "sites", ())}
    motif_bonds = tuple(tuple(sorted((int(i), int(j)))) for i, j in getattr(motif, "bonds", ()))
    if not motif_sites:
        return False
    if not motif_bonds:
        return bool(motif_sites & support_set)

    if str(motif_family).strip().lower() == "paop_bond_disp_std":
        for bond in motif_bonds:
            if set(bond).issubset(support_set):
                return True
            if bond in nearest_neighbor_bonds and bond[0] in support_set and bond[1] in support_set:
                return True
        return False

    return bool(motif_sites & support_set)


def _build_hh_uccsd_paop_product_pool(
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
    family_key: str,
    paop_r: int,
    paop_split_paulis: bool,
    paop_prune_eps: float,
    paop_normalization: str,
    num_particles: tuple[int, int],
) -> tuple[list[AnsatzTerm], dict[str, Any]]:
    del paop_r  # reserved for future locality extensions
    if make_phonon_motifs is None:
        raise RuntimeError(f"PAOP product pool requested but operator_pools module unavailable: {_PAOP_IMPORT_ERROR}")

    family_key_norm = str(family_key).strip().lower()
    spec = _HH_UCCSD_PAOP_PRODUCT_SPECS.get(family_key_norm)
    if spec is None:
        raise ValueError(f"Unsupported HH UCCSD⊗PAOP product family '{family_key}'.")
    if bool(paop_split_paulis):
        raise ValueError("UCCSD⊗PAOP product families do not support --paop-split-paulis; keep grouped logical generators intact.")

    motif_family = str(spec["motif_family"])
    parameterization = str(spec["parameterization"])
    seq2p = parameterization == "double_sequential"
    family_label_prefix = "uccsd_otimes_paop_seq2p" if seq2p else "uccsd_otimes_paop"

    uccsd_lifted_pool = _build_hh_uccsd_fermion_lifted_pool(
        int(num_sites),
        int(n_ph_max),
        str(boson_encoding),
        str(ordering),
        str(boundary),
        num_particles=tuple(num_particles),
    )
    motifs = make_phonon_motifs(
        motif_family,
        num_sites=int(num_sites),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        boundary=str(boundary),
        prune_eps=float(paop_prune_eps),
        normalization=str(paop_normalization),
    )
    nearest_neighbor_bonds = {
        tuple(sorted((int(i), int(j))))
        for i, j in bravais_nearest_neighbor_edges(
            int(num_sites),
            pbc=(str(boundary).strip().lower() == "periodic"),
        )
    }

    sorted_uccsd = sorted(
        list(uccsd_lifted_pool),
        key=lambda op: (
            0 if _parse_lifted_uccsd_support(str(op.label), num_sites=int(num_sites), ordering=str(ordering))[0] == "single" else 1,
            str(op.label),
        ),
    )
    ordered_motifs = sorted(list(motifs), key=lambda motif: (str(motif.family), str(motif.label)))

    raw_pool: list[AnsatzTerm] = []
    raw_pair_count = 0
    for op in sorted_uccsd:
        _kind, support_sites = _parse_lifted_uccsd_support(
            str(op.label),
            num_sites=int(num_sites),
            ordering=str(ordering),
        )
        for motif in ordered_motifs:
            if not _motif_matches_excitation_support(
                motif=motif,
                motif_family=motif_family,
                support_sites=support_sites,
                nearest_neighbor_bonds=nearest_neighbor_bonds,
            ):
                continue
            raw_pair_count += 1
            base_label = f"{family_label_prefix}::{op.label}::{motif.family}::{motif.label}"
            if seq2p:
                raw_pool.append(AnsatzTerm(label=f"{base_label}::step=ferm", polynomial=op.polynomial))
                raw_pool.append(AnsatzTerm(label=f"{base_label}::step=motif", polynomial=motif.poly))
                continue
            product_poly = _clean_real_pool_polynomial(op.polynomial * motif.poly, float(paop_prune_eps))
            if not product_poly.return_polynomial():
                continue
            raw_pool.append(AnsatzTerm(label=base_label, polynomial=product_poly))

    if seq2p:
        pool = list(raw_pool)
        dedup_strategy = "disabled_pair_label_preserving"
    elif int(n_ph_max) >= 2:
        pool = _deduplicate_pool_terms_lightweight(raw_pool)
        dedup_strategy = "signature_digest"
    else:
        pool = _deduplicate_pool_terms(raw_pool)
        dedup_strategy = "signature"

    return list(pool), {
        "family": family_key_norm,
        "family_kind": "uccsd_paop_product",
        "parameterization": parameterization,
        "motif_family": motif_family,
        "locality_rule": (
            "lf_overlap"
            if motif_family in {"paop_lf_std", "paop_lf2_std"}
            else "bond_disp_local_compatible"
        ),
        "raw_sizes": {
            "raw_uccsd_lifted": int(len(uccsd_lifted_pool)),
            "raw_phonon_motifs": int(len(motifs)),
            "raw_logical_pairs": int(raw_pair_count),
            "raw_emitted_terms": int(len(raw_pool)),
        },
        "logical_element_count": int(raw_pair_count),
        "expanded_term_count": int(len(pool)),
        "dedup_strategy": dedup_strategy,
        "dedup_total": int(len(pool)),
    }


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
    """Build HH full meta-pool: uccsd_lifted + hva + hh_termwise_augmented + paop_full + paop_lf_full."""
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


_HH_FULL_META_CLASSIFIER_VERSION = "hh_full_meta_v1"
_HH_FULL_META_ALLOWED_CLASSES = (
    "hh_termwise_unit",
    "hh_termwise_quadrature",
    "uccsd_sing",
    "uccsd_dbl",
    "hva_layer",
    "paop_cloud_p",
    "paop_cloud_x",
    "paop_disp",
    "paop_dbl",
    "paop_hopdrag",
    "paop_dbl_p",
    "paop_dbl_x",
    "paop_curdrag",
    "paop_hop2",
)


@dataclass(frozen=True)
class HHFullMetaClassFilterSpec:
    keep_classes: tuple[str, ...]
    classifier_version: str = _HH_FULL_META_CLASSIFIER_VERSION
    source_pool: str = "full_meta"
    source_problem: str = "hh"
    source_num_sites: int | None = None
    source_n_ph_max: int | None = None
    source_json: str | None = None


def _classify_hh_full_meta_label(label: str) -> str | None:
    """Map a deduplicated full_meta label onto a stable operator-class name."""
    label_str = str(label)
    if any(
        label_str.startswith(prefix)
        for prefix in ("hop_layer", "onsite_layer", "phonon_layer", "eph_layer")
    ):
        return "hva_layer"
    if label_str.startswith("hh_termwise_ham_unit_term("):
        return "hh_termwise_unit"
    if label_str.startswith("hh_termwise_ham_quadrature_term("):
        return "hh_termwise_quadrature"
    if label_str.startswith("uccsd_ferm_lifted::uccsd_sing("):
        return "uccsd_sing"
    if label_str.startswith("uccsd_ferm_lifted::uccsd_dbl("):
        return "uccsd_dbl"
    if label_str.startswith("paop_full:paop_cloud_p("):
        return "paop_cloud_p"
    if label_str.startswith("paop_full:paop_cloud_x("):
        return "paop_cloud_x"
    if label_str.startswith("paop_full:paop_disp("):
        return "paop_disp"
    if label_str.startswith("paop_full:paop_dbl("):
        return "paop_dbl"
    if label_str.startswith("paop_full:paop_hopdrag("):
        return "paop_hopdrag"
    if label_str.startswith("paop_lf_full:paop_dbl_p("):
        return "paop_dbl_p"
    if label_str.startswith("paop_lf_full:paop_dbl_x("):
        return "paop_dbl_x"
    if label_str.startswith("paop_lf_full:paop_curdrag("):
        return "paop_curdrag"
    if label_str.startswith("paop_lf_full:paop_hop2("):
        return "paop_hop2"
    return None


def _normalize_hh_full_meta_keep_classes(classes: Sequence[Any]) -> tuple[str, ...]:
    keep_classes: list[str] = []
    seen: set[str] = set()
    for raw in classes:
        name = str(raw).strip()
        if name == "":
            continue
        if name not in _HH_FULL_META_ALLOWED_CLASSES:
            raise ValueError(
                "Unknown HH full_meta class "
                f"{name!r}; allowed classes are {list(_HH_FULL_META_ALLOWED_CLASSES)}."
            )
        if name in seen:
            continue
        seen.add(name)
        keep_classes.append(name)
    if not keep_classes:
        raise ValueError("HH full_meta class filter must keep at least one class.")
    return tuple(keep_classes)


def _load_hh_full_meta_class_filter_spec(path: Path) -> HHFullMetaClassFilterSpec:
    """Load a keep-spec JSON for filtering the HH full_meta pool by class."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("HH full_meta class filter JSON must be an object with keep_classes.")
    keep_raw = raw.get("keep_classes")
    if not isinstance(keep_raw, list):
        raise ValueError("HH full_meta class filter JSON must contain list field 'keep_classes'.")
    classifier_version = str(raw.get("classifier_version", _HH_FULL_META_CLASSIFIER_VERSION)).strip()
    if classifier_version != _HH_FULL_META_CLASSIFIER_VERSION:
        raise ValueError(
            "HH full_meta class filter classifier_version mismatch: "
            f"got {classifier_version!r}, expected {_HH_FULL_META_CLASSIFIER_VERSION!r}."
        )
    source_pool = str(raw.get("source_pool", "")).strip().lower()
    if source_pool != "full_meta":
        raise ValueError(
            "HH full_meta class filter JSON must declare source_pool='full_meta'."
        )
    source_problem = str(raw.get("source_problem", "hh")).strip().lower()
    if source_problem != "hh":
        raise ValueError(
            "HH full_meta class filter JSON must declare source_problem='hh'."
        )
    source_num_sites_raw = raw.get("source_num_sites")
    source_n_ph_max_raw = raw.get("source_n_ph_max")
    return HHFullMetaClassFilterSpec(
        keep_classes=_normalize_hh_full_meta_keep_classes(keep_raw),
        classifier_version=str(classifier_version),
        source_pool=str(source_pool),
        source_problem=str(source_problem),
        source_num_sites=(
            None if source_num_sites_raw is None else int(source_num_sites_raw)
        ),
        source_n_ph_max=(
            None if source_n_ph_max_raw is None else int(source_n_ph_max_raw)
        ),
        source_json=str(path),
    )


def _summarize_hh_full_meta_pool_classes(pool: Sequence[AnsatzTerm]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for term in pool:
        family = _classify_hh_full_meta_label(str(term.label))
        if family is None:
            raise ValueError(f"Unable to classify HH full_meta operator label {term.label!r}.")
        counts[family] = int(counts.get(family, 0) + 1)
    return {
        family: int(counts[family])
        for family in _HH_FULL_META_ALLOWED_CLASSES
        if family in counts
    }


def _filter_hh_full_meta_pool_by_class(
    pool: Sequence[AnsatzTerm],
    spec: HHFullMetaClassFilterSpec,
) -> tuple[list[AnsatzTerm], dict[str, Any]]:
    """Filter a deduplicated HH full_meta pool down to the requested classes."""
    counts_before = _summarize_hh_full_meta_pool_classes(pool)
    keep_set = set(spec.keep_classes)
    filtered_pool = [
        term
        for term in pool
        if _classify_hh_full_meta_label(str(term.label)) in keep_set
    ]
    if not filtered_pool:
        raise ValueError("HH full_meta class filter removed every operator from the pool.")
    counts_after = _summarize_hh_full_meta_pool_classes(filtered_pool)
    dropped_classes = [
        family for family in counts_before.keys()
        if family not in keep_set
    ]
    meta = {
        "classifier_version": str(spec.classifier_version),
        "source_pool": str(spec.source_pool),
        "source_problem": str(spec.source_problem),
        "source_num_sites": (
            int(spec.source_num_sites) if spec.source_num_sites is not None else None
        ),
        "source_n_ph_max": (
            int(spec.source_n_ph_max) if spec.source_n_ph_max is not None else None
        ),
        "source_json": str(spec.source_json) if spec.source_json is not None else None,
        "keep_classes": [str(x) for x in spec.keep_classes],
        "dropped_classes": [str(x) for x in dropped_classes],
        "class_counts_before": dict(counts_before),
        "class_counts_after": dict(counts_after),
        "dedup_total_before": int(len(pool)),
        "dedup_total_after": int(len(filtered_pool)),
    }
    return filtered_pool, meta


# ---------------------------------------------------------------------------
# Pareto-lean pool: pruned subset of full_meta informed by motif analysis.
# Retains only operator classes that contributed energy improvement in the
# best-yet heavy scaffold run.  See artifacts/reports/hh_heavy_scaffold_best_yet_20260321.md
# ---------------------------------------------------------------------------

_PARETO_LEAN_PAOP_FULL_KEEP = {"paop_cloud_p", "paop_disp", "paop_hopdrag"}
_PARETO_LEAN_PAOP_LF_KEEP = {"paop_dbl_p"}


def _pareto_lean_paop_match(label: str, allowed: set[str]) -> bool:
    """True if the PAOP label's family name is in *allowed*."""
    colon_idx = label.find(":")
    if colon_idx < 0:
        return False
    after_colon = label[colon_idx + 1:]
    for family in allowed:
        if after_colon.startswith(family + "("):
            return True
    return False


def _build_hh_pareto_lean_pool(
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
    """Build the Pareto-lean HH pool: only classes that survived the heavy scaffold.

    Kept classes:
      - uccsd_sing, uccsd_dbl  (all lifted UCCSD)
      - hh_termwise_quadrature (y-partner quadrature terms only, no unit terms)
      - paop_cloud_p, paop_disp, paop_hopdrag  (from paop_full)
      - paop_dbl_p  (from paop_lf_full)

    Dropped classes:
      - HVA layerwise macros  (hop_layer, onsite_layer, phonon_layer, eph_layer)
      - hh_termwise_unit       (diagonal Ham terms, never selected)
      - paop_cloud_x, paop_dbl, paop_lf_dbl_x, paop_lf_curdrag, paop_lf_hop2
    """
    # 1. Lifted UCCSD (all singles + doubles)
    uccsd_lifted_pool = _build_hh_uccsd_fermion_lifted_pool(
        int(num_sites),
        int(n_ph_max),
        str(boson_encoding),
        str(ordering),
        str(boundary),
        num_particles=num_particles,
    )

    # 2. Termwise quadrature only (skip unit terms)
    quadrature_pool: list[AnsatzTerm] = []
    if abs(float(g_ep)) > 1e-15:
        termwise_aug = _build_hh_termwise_augmented_pool(h_poly)
        quadrature_pool = [
            AnsatzTerm(label=f"hh_termwise_{term.label}", polynomial=term.polynomial)
            for term in termwise_aug
            if "quadrature" in term.label
        ]

    # 3. PAOP full, filtered to cloud_p + disp + hopdrag
    paop_full_raw = _build_paop_pool(
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
    paop_full_kept = [
        t for t in paop_full_raw
        if _pareto_lean_paop_match(t.label, _PARETO_LEAN_PAOP_FULL_KEEP)
    ]

    # 4. PAOP lf_full, filtered to dbl_p only
    paop_lf_raw = _build_paop_pool(
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
    paop_lf_kept = [
        t for t in paop_lf_raw
        if _pareto_lean_paop_match(t.label, _PARETO_LEAN_PAOP_LF_KEEP)
    ]

    merged = (
        list(uccsd_lifted_pool)
        + list(quadrature_pool)
        + list(paop_full_kept)
        + list(paop_lf_kept)
    )
    meta = {
        "raw_uccsd_lifted": int(len(uccsd_lifted_pool)),
        "raw_hh_termwise_quadrature": int(len(quadrature_pool)),
        "raw_paop_full_kept": int(len(paop_full_kept)),
        "raw_paop_full_dropped": int(len(paop_full_raw) - len(paop_full_kept)),
        "raw_paop_lf_kept": int(len(paop_lf_kept)),
        "raw_paop_lf_dropped": int(len(paop_lf_raw) - len(paop_lf_kept)),
        "raw_total": int(len(merged)),
    }
    if int(n_ph_max) >= 2:
        dedup_pool = _deduplicate_pool_terms_lightweight(merged)
    else:
        dedup_pool = _deduplicate_pool_terms(merged)
    return dedup_pool, meta


# ---------------------------------------------------------------------------
# Pareto-lean L3 pool: explicit public alias for the historical L=3
# class-pruned family recovered from the heavy full_meta report.
# This stays class-level only; no site-specific keep/drop logic.
# ---------------------------------------------------------------------------

def _build_hh_pareto_lean_l3_pool(
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
    """Build the L=3-specific class-pruned Pareto-lean pool.

    This is the explicit public L=3 alias of the historical heavy-run
    class family:
      - uccsd_sing, uccsd_dbl
      - hh_termwise_quadrature
      - paop_cloud_p, paop_disp, paop_hopdrag
      - paop_dbl_p

    It intentionally does not apply any site-specific keep/drop rules.
    """
    if int(num_sites) != 3:
        raise ValueError("adapt_pool='pareto_lean_l3' is only valid for L=3.")
    if int(n_ph_max) != 1:
        raise ValueError("adapt_pool='pareto_lean_l3' is only valid for n_ph_max=1.")
    return _build_hh_pareto_lean_pool(
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


# ---------------------------------------------------------------------------
# Pareto-lean L2 pool: tighter pruning from L=2 n_ph_max=1 motif analysis.
# Drops disp, uccsd_dbl, and unused dbl_p site-phonon pairings.
# See artifacts/reports/hh_L2_ecut1_scaffold_motif_analysis.md
# ---------------------------------------------------------------------------

_PARETO_LEAN_L2_PAOP_FULL_KEEP = {"paop_cloud_p", "paop_hopdrag"}
_PARETO_LEAN_L2_PAOP_LF_KEEP = {"paop_dbl_p"}

# Only the site->phonon pairings that were actually selected at L=2.
# site=0->phonon=1 and site=1->phonon=0 were used; the others were not.
_PARETO_LEAN_L2_DPL_P_KEEP_SUFFIXES = {"site=0->phonon=1", "site=1->phonon=0"}


def _build_hh_pareto_lean_l2_pool(
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
    """Build the L=2-specific Pareto-lean pool (11 operators).

    Kept: quadrature(4), uccsd_sing(all), cloud_p(all), hopdrag(all),
          dbl_p(site=0->ph=1, site=1->ph=0 only).
    Dropped: uccsd_dbl, disp, dbl (bare), all x-type, curdrag, hop2,
             HVA layers, unit terms, unused dbl_p variants.
    """
    if int(num_sites) != 2:
        raise ValueError("adapt_pool='pareto_lean_l2' is only valid for L=2.")
    if int(n_ph_max) != 1:
        raise ValueError("adapt_pool='pareto_lean_l2' is only valid for n_ph_max=1.")
    # 1. UCCSD lifted — singles only
    all_uccsd = _build_hh_uccsd_fermion_lifted_pool(
        int(num_sites),
        int(n_ph_max),
        str(boson_encoding),
        str(ordering),
        str(boundary),
        num_particles=num_particles,
    )
    uccsd_singles = [t for t in all_uccsd if "uccsd_sing" in t.label]

    # 2. Quadrature only
    quadrature_pool: list[AnsatzTerm] = []
    if abs(float(g_ep)) > 1e-15:
        termwise_aug = _build_hh_termwise_augmented_pool(h_poly)
        quadrature_pool = [
            AnsatzTerm(label=f"hh_termwise_{term.label}", polynomial=term.polynomial)
            for term in termwise_aug
            if "quadrature" in term.label
        ]

    # 3. PAOP full: cloud_p + hopdrag only (no disp)
    paop_full_raw = _build_paop_pool(
        int(num_sites), int(n_ph_max), str(boson_encoding), str(ordering),
        str(boundary), "paop_full", int(paop_r), bool(paop_split_paulis),
        float(paop_prune_eps), str(paop_normalization), num_particles,
    )
    paop_full_kept = [
        t for t in paop_full_raw
        if _pareto_lean_paop_match(t.label, _PARETO_LEAN_L2_PAOP_FULL_KEEP)
    ]

    # 4. PAOP lf_full: dbl_p, filtered to used site->phonon pairings
    paop_lf_raw = _build_paop_pool(
        int(num_sites), int(n_ph_max), str(boson_encoding), str(ordering),
        str(boundary), "paop_lf_full", int(paop_r), bool(paop_split_paulis),
        float(paop_prune_eps), str(paop_normalization), num_particles,
    )
    paop_lf_kept = []
    for t in paop_lf_raw:
        if not _pareto_lean_paop_match(t.label, _PARETO_LEAN_L2_PAOP_LF_KEEP):
            continue
        # Further filter to only the used site->phonon pairings
        if any(suffix in t.label for suffix in _PARETO_LEAN_L2_DPL_P_KEEP_SUFFIXES):
            paop_lf_kept.append(t)

    merged = (
        list(uccsd_singles)
        + list(quadrature_pool)
        + list(paop_full_kept)
        + list(paop_lf_kept)
    )
    meta = {
        "raw_uccsd_singles": int(len(uccsd_singles)),
        "raw_hh_termwise_quadrature": int(len(quadrature_pool)),
        "raw_paop_full_kept": int(len(paop_full_kept)),
        "raw_paop_full_dropped": int(len(paop_full_raw) - len(paop_full_kept)),
        "raw_paop_lf_kept": int(len(paop_lf_kept)),
        "raw_paop_lf_dropped": int(len(paop_lf_raw) - len(paop_lf_kept)),
        "raw_total": int(len(merged)),
    }
    if int(n_ph_max) >= 2:
        dedup_pool = _deduplicate_pool_terms_lightweight(merged)
    else:
        dedup_pool = _deduplicate_pool_terms(merged)
    return dedup_pool, meta


# ---------------------------------------------------------------------------
# Gate-pruned pool: informed by term-level leave-one-out analysis.
# For operators where one Pauli term suffices, replace the multi-term
# polynomial with the single dominant term.  This halves the 2Q gate count
# for uccsd_sing and paop_hopdrag at zero energy regression.
# See artifacts/reports/hh_prune_nighthawk_pareto_menu_20260322.md
# ---------------------------------------------------------------------------

# Mapping: operator label pattern -> list of Pauli strings to KEEP.
# If an operator label matches, its polynomial is replaced with only the
# listed Pauli terms (preserving original coefficients).
# None means keep the original (no pruning).
_GATE_PRUNE_TERM_KEEP: dict[str, list[str] | None] = {
    "uccsd_sing(alpha:0->1)": ["eeeexy"],   # drop eeeeyx (reg ~0)
    "uccsd_sing(beta:2->3)":  ["eeyxee"],   # drop eexyee (reg ~0)
    "paop_hopdrag":           ["yeyyee"],    # drop yexxee (reg ~0)
}


def _gate_prune_polynomial(
    label: str,
    poly: Any,
) -> Any:
    """If the operator matches a gate-prune rule, return a trimmed polynomial."""
    for pattern, keep_paulis in _GATE_PRUNE_TERM_KEEP.items():
        if keep_paulis is None:
            continue
        if pattern in label:
            terms = poly.return_polynomial()
            if not terms:
                return poly
            nq = int(terms[0].nqubit())
            keep_set = set(keep_paulis)
            kept_terms = [t for t in terms if str(t.pw2strng()) in keep_set]
            if not kept_terms:
                return poly  # safety: if nothing matches, keep original
            pruned = PauliPolynomial("JW", [
                PauliTerm(nq, ps=str(t.pw2strng()), pc=float(t.p_coeff))
                for t in kept_terms
            ])
            return pruned
    return poly


def _build_hh_pareto_lean_gate_pruned_pool(
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
    """Pareto-lean pool with gate-level term pruning.

    Starts from pareto_lean, then replaces multi-term operators with single-term
    variants where term-level leave-one-out showed zero regression:
      - uccsd_sing(alpha:0->1): keep only eeeexy (drop eeeeyx)
      - uccsd_sing(beta:2->3):  keep only eeyxee (drop eexyee)
      - paop_hopdrag(*):        keep only yeyyee (drop yexxee)

    All other operators (paop_dbl_p, paop_disp, quadrature) are unchanged.
    """
    # Build the base pareto_lean pool
    base_pool, base_meta = _build_hh_pareto_lean_pool(
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

    # Apply gate-level term pruning
    pruned_pool: list[AnsatzTerm] = []
    n_pruned = 0
    for term in base_pool:
        pruned_poly = _gate_prune_polynomial(term.label, term.polynomial)
        if pruned_poly is not term.polynomial:
            n_pruned += 1
        pruned_pool.append(AnsatzTerm(label=term.label, polynomial=pruned_poly))

    meta = dict(base_meta)
    meta["gate_pruned_operators"] = int(n_pruned)
    meta["gate_prune_rules"] = {k: v for k, v in _GATE_PRUNE_TERM_KEEP.items() if v is not None}
    return pruned_pool, meta


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
    *,
    parameter_layout: AnsatzParameterLayout | None = None,
) -> np.ndarray:
    """Apply the current ADAPT ansatz.

    Supports both legacy logical-shared theta and per-Pauli runtime theta.
    """
    psi = np.array(psi_ref, copy=True)
    if not selected_ops:
        return psi
    theta_arr = np.asarray(theta, dtype=float).reshape(-1)
    layout = (
        parameter_layout
        if parameter_layout is not None
        else build_parameter_layout(selected_ops, ignore_identity=True, coefficient_tolerance=1e-12, sort_terms=True)
    )
    if int(theta_arr.size) == int(layout.runtime_parameter_count):
        for block, op in zip(layout.blocks, selected_ops):
            if int(block.runtime_count) <= 0:
                continue
            psi = apply_exp_pauli_polynomial_termwise(
                psi,
                op.polynomial,
                theta_arr[block.runtime_start:block.runtime_stop],
                ignore_identity=bool(layout.ignore_identity),
                coefficient_tolerance=float(layout.coefficient_tolerance),
                sort_terms=(str(layout.term_order) == "sorted"),
            )
        return psi
    if int(theta_arr.size) == int(len(selected_ops)):
        for k, op in enumerate(selected_ops):
            psi = apply_exp_pauli_polynomial(psi, op.polynomial, float(theta_arr[k]))
        return psi
    raise ValueError(
        f"ADAPT theta length mismatch: got {theta_arr.size}, expected {layout.runtime_parameter_count} (runtime) or {len(selected_ops)} (logical)."
    )


def _logical_theta_alias(theta: np.ndarray, layout: AnsatzParameterLayout) -> np.ndarray:
    theta_arr = np.asarray(theta, dtype=float).reshape(-1)
    if int(theta_arr.size) == int(layout.runtime_parameter_count):
        return np.asarray(project_runtime_theta_block_mean(theta_arr, layout), dtype=float)
    if int(theta_arr.size) == int(layout.logical_parameter_count):
        return np.asarray(theta_arr, dtype=float)
    raise ValueError(
        f"Cannot project theta of length {theta_arr.size} onto logical blocks of size {layout.logical_parameter_count}."
    )


def _adapt_energy_fn(
    h_poly: Any,
    psi_ref: np.ndarray,
    selected_ops: list[AnsatzTerm],
    theta: np.ndarray,
    *,
    h_compiled: CompiledPolynomialAction | None = None,
    parameter_layout: AnsatzParameterLayout | None = None,
) -> float:
    """Energy of the current ADAPT ansatz at parameters theta."""
    psi = _prepare_adapt_state(psi_ref, selected_ops, theta, parameter_layout=parameter_layout)
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
_VALID_ADAPT_INNER_OPTIMIZERS = frozenset({"COBYLA", "POWELL", "SPSA"})


def _scipy_adapt_heartbeat_event(method_key: str) -> str:
    method = str(method_key).strip().upper()
    if method == "COBYLA":
        return "hardcoded_adapt_cobyla_heartbeat"
    return "hardcoded_adapt_scipy_heartbeat"


def _scipy_adapt_optimizer_options(*, method_key: str, maxiter: int) -> dict[str, Any]:
    method = str(method_key).strip().upper()
    options: dict[str, Any] = {"maxiter": int(maxiter)}
    if method == "COBYLA":
        options["rhobeg"] = 0.3
    return options


def _run_scipy_adapt_optimizer(
    *,
    method_key: str,
    objective: Any,
    x0: np.ndarray,
    maxiter: int,
    context_label: str,
    scipy_minimize_fn: Any,
) -> Any:
    if scipy_minimize_fn is None:
        raise RuntimeError(f"SciPy minimize is unavailable for {method_key} {context_label}.")
    return scipy_minimize_fn(
        objective,
        x0,
        method=str(method_key),
        options=_scipy_adapt_optimizer_options(method_key=str(method_key), maxiter=int(maxiter)),
    )


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
    problem_key = str(problem).strip().lower()
    if requested_mode is None:
        return "phase3_v1" if problem_key == "hh" else "legacy"
    mode_raw = str(requested_mode).strip().lower()
    if mode_raw == "":
        return "phase3_v1" if problem_key == "hh" else "legacy"
    if mode_raw not in {"legacy", "phase1_v1", "phase2_v1", "phase3_v1"}:
        raise ValueError("adapt_continuation_mode must be one of {'legacy','phase1_v1','phase2_v1','phase3_v1'}.")
    return str(mode_raw)


def _resolve_cli_adapt_continuation_mode(*, problem: str, requested_mode: str | None) -> str:
    return _resolve_adapt_continuation_mode(problem=problem, requested_mode=requested_mode)


@dataclass(frozen=True)
class Phase3OracleGradientConfig:
    noise_mode: str
    shots: int
    oracle_repeats: int
    oracle_aggregate: str
    backend_name: str | None
    use_fake_backend: bool
    seed: int
    gradient_step: float
    mitigation_mode: str
    local_readout_strategy: str | None
    zne_scales: tuple[float, ...] = ()
    local_gate_twirling: bool = False
    dd_sequence: str | None = None
    scope: str = "selection_only"
    execution_surface_requested: str = "auto"
    execution_surface: str = "expectation_v1"
    raw_transport: str = "auto"
    raw_store_memory: bool = False
    raw_artifact_path: str | None = None
    seed_transpiler: int | None = None
    transpile_optimization_level: int = 1


_FINAL_NOISE_AUDIT_RUNTIME_PROFILE_NAMES = {
    "legacy_runtime_v0",
    "main_twirled_readout_v1",
    "dd_probe_twirled_readout_v1",
    "final_audit_zne_twirled_readout_v1",
}

_FINAL_NOISE_AUDIT_RUNTIME_SESSION_POLICIES = {
    "prefer_session",
    "require_session",
    "backend_only",
}


@dataclass(frozen=True)
class FinalNoiseAuditConfig:
    noise_mode: str
    shots: int
    oracle_repeats: int
    oracle_aggregate: str
    backend_name: str | None
    use_fake_backend: bool
    seed: int
    mitigation_mode: str
    local_readout_strategy: str | None
    zne_scales: tuple[float, ...] = ()
    local_gate_twirling: bool = False
    dd_sequence: str | None = None
    runtime_profile_name: str = "legacy_runtime_v0"
    runtime_session_policy: str = "prefer_session"
    compare_unmitigated_baseline: bool = False
    seed_transpiler: int | None = None
    transpile_optimization_level: int = 1
    strict: bool = False


@dataclass(frozen=True)
class FinalNoiseAuditSnapshot:
    h_poly: Any
    parameter_layout: AnsatzParameterLayout
    theta_runtime: tuple[float, ...]
    theta_logical: tuple[float, ...]
    reference_state: np.ndarray
    num_qubits: int
    operator_labels: tuple[str, ...]
    ansatz_depth: int
    runtime_parameter_count: int
    logical_parameter_count: int
    exact_filtered_ground_energy: float
    exact_final_state_energy: float


def _parse_oracle_zne_scales(
    raw: Any,
    *,
    field_name: str,
) -> tuple[float, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        tokens = [tok.strip() for tok in str(raw).split(",") if tok.strip() != ""]
    elif isinstance(raw, Sequence):
        tokens = list(raw)
    else:
        tokens = [raw]
    out: list[float] = []
    for tok in tokens:
        value = float(tok)
        if (not math.isfinite(value)) or value <= 0.0:
            raise ValueError(f"{field_name} entries must be finite and > 0.")
        out.append(float(value))
    return tuple(out)


def _validate_backend_scheduled_local_zne_scales(
    zne_scales: Sequence[float],
    *,
    field_name: str,
) -> tuple[float, ...]:
    out: list[float] = []
    for raw in zne_scales:
        value = float(raw)
        rounded = int(round(value))
        if (
            (not math.isfinite(value))
            or rounded < 1
            or rounded % 2 == 0
            or (not math.isclose(value, float(rounded), rel_tol=0.0, abs_tol=1e-9))
        ):
            raise ValueError(
                f"{field_name} must contain odd positive integer noise scales for backend_scheduled local ZNE."
            )
        out.append(float(rounded))
    if out and not any(math.isclose(val, 1.0, rel_tol=0.0, abs_tol=1e-9) for val in out):
        raise ValueError(
            f"{field_name} must include the base noise scale 1 for backend_scheduled local ZNE."
        )
    return tuple(out)


def _resolve_phase3_oracle_gradient_config(
    config: Phase3OracleGradientConfig,
) -> Phase3OracleGradientConfig:
    requested_surface = str(getattr(config, "execution_surface_requested", "auto")).strip().lower() or "auto"
    if requested_surface not in {"auto", "expectation_v1", "raw_measurement_v1"}:
        raise ValueError(
            "phase3_oracle_execution_surface must be one of {'auto','expectation_v1','raw_measurement_v1'}."
        )
    noise_mode = str(config.noise_mode).strip().lower()
    mitigation_mode = str(config.mitigation_mode).strip().lower()
    resolved_surface = (
        "raw_measurement_v1"
        if requested_surface == "auto"
        and noise_mode == "runtime"
        and mitigation_mode == "none"
        else (
            "expectation_v1"
            if requested_surface == "auto"
            else str(requested_surface)
        )
    )
    return Phase3OracleGradientConfig(
        noise_mode=str(noise_mode),
        shots=int(config.shots),
        oracle_repeats=int(config.oracle_repeats),
        oracle_aggregate=str(config.oracle_aggregate).strip().lower(),
        backend_name=(None if config.backend_name in {None, ""} else str(config.backend_name)),
        use_fake_backend=bool(config.use_fake_backend),
        seed=int(config.seed),
        gradient_step=float(config.gradient_step),
        mitigation_mode=str(mitigation_mode),
        local_readout_strategy=(
            None
            if config.local_readout_strategy in {None, ""}
            else str(config.local_readout_strategy).strip().lower()
        ),
        zne_scales=_parse_oracle_zne_scales(
            getattr(config, "zne_scales", ()),
            field_name="phase3_oracle_zne_scales",
        ),
        local_gate_twirling=bool(getattr(config, "local_gate_twirling", False)),
        dd_sequence=(
            None
            if getattr(config, "dd_sequence", None) in {None, "", "none"}
            else str(getattr(config, "dd_sequence")).strip()
        ),
        scope=str(config.scope).strip().lower() or "selection_only",
        execution_surface_requested=str(requested_surface),
        execution_surface=str(resolved_surface),
        raw_transport=str(getattr(config, "raw_transport", "auto")).strip().lower() or "auto",
        raw_store_memory=bool(getattr(config, "raw_store_memory", False)),
        raw_artifact_path=(
            None
            if getattr(config, "raw_artifact_path", None) in {None, ""}
            else str(getattr(config, "raw_artifact_path"))
        ),
        seed_transpiler=(
            None
            if getattr(config, "seed_transpiler", None) is None
            else int(getattr(config, "seed_transpiler"))
        ),
        transpile_optimization_level=int(getattr(config, "transpile_optimization_level", 1)),
    )


def _resolve_final_noise_audit_config(
    config: FinalNoiseAuditConfig,
) -> FinalNoiseAuditConfig:
    return FinalNoiseAuditConfig(
        noise_mode=str(config.noise_mode).strip().lower(),
        shots=int(config.shots),
        oracle_repeats=int(config.oracle_repeats),
        oracle_aggregate=str(config.oracle_aggregate).strip().lower(),
        backend_name=(None if config.backend_name in {None, ""} else str(config.backend_name)),
        use_fake_backend=bool(config.use_fake_backend),
        seed=int(config.seed),
        mitigation_mode=str(config.mitigation_mode).strip().lower(),
        local_readout_strategy=(
            None
            if config.local_readout_strategy in {None, ""}
            else str(config.local_readout_strategy).strip().lower()
        ),
        zne_scales=_parse_oracle_zne_scales(
            getattr(config, "zne_scales", ()),
            field_name="final_noise_audit_zne_scales",
        ),
        local_gate_twirling=bool(getattr(config, "local_gate_twirling", False)),
        dd_sequence=(
            None
            if getattr(config, "dd_sequence", None) in {None, "", "none"}
            else str(getattr(config, "dd_sequence")).strip()
        ),
        runtime_profile_name=(
            str(getattr(config, "runtime_profile_name", "legacy_runtime_v0")).strip().lower()
            or "legacy_runtime_v0"
        ),
        runtime_session_policy=(
            str(getattr(config, "runtime_session_policy", "prefer_session")).strip().lower()
            or "prefer_session"
        ),
        compare_unmitigated_baseline=bool(
            getattr(config, "compare_unmitigated_baseline", False)
        ),
        seed_transpiler=(
            None if config.seed_transpiler is None else int(config.seed_transpiler)
        ),
        transpile_optimization_level=int(config.transpile_optimization_level),
        strict=bool(config.strict),
    )


def _json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _phase3_oracle_runtime_bindings() -> dict[str, Any]:
    from pipelines.exact_bench.noise_oracle_runtime import (
        ExpectationOracle,
        OracleConfig,
        RawMeasurementOracle,
        _all_z_full_register_qop,
        _summarize_hh_full_register_z_records,
        assess_oracle_execution_capability,
        build_runtime_layout_circuit,
        normalize_oracle_execution_request,
        normalize_sampler_raw_runtime_config,
        pauli_poly_to_sparse_pauli_op,
        preflight_backend_scheduled_fake_backend_environment,
        validate_oracle_execution_request,
    )
    from pipelines.hardcoded.adapt_circuit_execution import build_parameterized_ansatz_plan

    return {
        "ExpectationOracle": ExpectationOracle,
        "OracleConfig": OracleConfig,
        "RawMeasurementOracle": RawMeasurementOracle,
        "assess_oracle_execution_capability": assess_oracle_execution_capability,
        "all_z_full_register_qop": _all_z_full_register_qop,
        "summarize_hh_full_register_z_records": _summarize_hh_full_register_z_records,
        "build_runtime_layout_circuit": build_runtime_layout_circuit,
        "build_parameterized_ansatz_plan": build_parameterized_ansatz_plan,
        "normalize_oracle_execution_request": normalize_oracle_execution_request,
        "normalize_sampler_raw_runtime_config": normalize_sampler_raw_runtime_config,
        "pauli_poly_to_sparse_pauli_op": pauli_poly_to_sparse_pauli_op,
        "preflight_backend_scheduled_fake_backend_environment": preflight_backend_scheduled_fake_backend_environment,
        "validate_oracle_execution_request": validate_oracle_execution_request,
    }


def _validate_oracle_execution_request_via_bindings(
    bindings: Mapping[str, Any],
    oracle_config: Any,
) -> dict[str, Any] | None:
    validate_fn = bindings.get("validate_oracle_execution_request")
    if callable(validate_fn):
        return _json_ready(validate_fn(oracle_config))
    fallback_validate_fn = bindings.get("validate_controller_oracle_base_config")
    if callable(fallback_validate_fn):
        fallback_validate_fn(oracle_config)
    normalize_fn = bindings.get("normalize_oracle_execution_request")
    if callable(normalize_fn):
        return {
            "supported": True,
            "reason_code": "ok",
            "reason": "ok",
            "normalized_request": _json_ready(normalize_fn(oracle_config)),
        }
    return None


def _validate_phase3_oracle_gradient_config(
    *,
    config: Phase3OracleGradientConfig,
    problem: str,
    continuation_mode: str,
) -> None:
    config = _resolve_phase3_oracle_gradient_config(config)
    problem_key = str(problem).strip().lower()
    continuation_key = str(continuation_mode).strip().lower()
    if problem_key != "hh":
        raise ValueError("phase3 oracle gradient mode is only valid for problem='hh'.")
    if continuation_key != "phase3_v1":
        raise ValueError(
            "phase3 oracle gradient mode is only valid for adapt_continuation_mode='phase3_v1'."
        )
    noise_mode = str(config.noise_mode).strip().lower()
    if noise_mode not in {"ideal", "shots", "aer_noise", "backend_scheduled", "runtime"}:
        raise ValueError(
            "phase3_oracle_gradient_mode must be one of {'off','ideal','shots','aer_noise','backend_scheduled','runtime'}."
        )
    if int(config.shots) < 1:
        raise ValueError("phase3_oracle_shots must be >= 1.")
    if int(config.oracle_repeats) < 1:
        raise ValueError("phase3_oracle_repeats must be >= 1.")
    aggregate_key = str(config.oracle_aggregate).strip().lower()
    if aggregate_key != "mean":
        raise ValueError("phase3 oracle gradient mode currently requires oracle_aggregate='mean'.")
    if (not math.isfinite(float(config.gradient_step))) or float(config.gradient_step) <= 0.0:
        raise ValueError("phase3_oracle_gradient_step must be finite and > 0.")
    mitigation_mode = str(config.mitigation_mode).strip().lower()
    if mitigation_mode not in {"none", "readout"}:
        raise ValueError("phase3_oracle_mitigation must be one of {'none','readout'}.")
    zne_scales = tuple(float(x) for x in getattr(config, "zne_scales", ()) or ())
    local_gate_twirling = bool(getattr(config, "local_gate_twirling", False))
    dd_sequence = (
        None
        if getattr(config, "dd_sequence", None) in {None, "", "none"}
        else str(getattr(config, "dd_sequence")).strip()
    )
    if noise_mode == "backend_scheduled" and not bool(config.use_fake_backend):
        raise ValueError(
            "phase3 oracle gradient mode backend_scheduled requires --phase3-oracle-use-fake-backend."
        )
    if noise_mode == "runtime" and config.backend_name in {None, ""}:
        raise ValueError("phase3 oracle gradient runtime mode requires --phase3-oracle-backend-name.")
    local_readout_strategy = (
        None
        if config.local_readout_strategy in {None, ""}
        else str(config.local_readout_strategy).strip().lower()
    )
    if mitigation_mode == "readout" and local_readout_strategy not in {None, "mthree"}:
        raise ValueError(
            "phase3_oracle_local_readout_strategy must be 'mthree' when readout mitigation is enabled."
        )
    if mitigation_mode != "readout" and local_readout_strategy is not None:
        raise ValueError(
            "phase3_oracle_local_readout_strategy requires phase3_oracle_mitigation='readout'."
        )
    if noise_mode != "backend_scheduled" and (
        zne_scales
        or local_gate_twirling
        or dd_sequence not in {None, "", "none"}
    ):
        raise ValueError(
            "phase3 oracle local ZNE/gate twirling/DD currently require noise_mode='backend_scheduled'."
        )
    if noise_mode == "backend_scheduled" and zne_scales:
        _validate_backend_scheduled_local_zne_scales(
            zne_scales,
            field_name="phase3_oracle_zne_scales",
        )
    if str(config.scope).strip().lower() != "selection_only":
        raise ValueError("phase3 oracle gradient scope is fixed to 'selection_only' in v1.")
    execution_surface = str(config.execution_surface).strip().lower()
    if execution_surface == "expectation_v1":
        return
    if execution_surface != "raw_measurement_v1":
        raise ValueError(
            "phase3_oracle_execution_surface must resolve to 'expectation_v1' or 'raw_measurement_v1'."
        )
    if mitigation_mode != "none":
        raise ValueError("phase3 raw oracle execution requires mitigation_mode='none'.")
    if local_readout_strategy is not None:
        raise ValueError("phase3 raw oracle execution does not allow local readout strategy.")
    if zne_scales:
        raise ValueError("phase3 raw oracle execution does not allow local ZNE scales.")
    if local_gate_twirling:
        raise ValueError("phase3 raw oracle execution does not allow local gate twirling.")
    if dd_sequence not in {None, "", "none"}:
        raise ValueError("phase3 raw oracle execution does not allow local DD.")
    raw_transport = str(config.raw_transport).strip().lower()
    if noise_mode == "backend_scheduled":
        if not bool(config.use_fake_backend):
            raise ValueError(
                "phase3 raw oracle execution requires --phase3-oracle-use-fake-backend when noise_mode='backend_scheduled'."
            )
        if raw_transport != "auto":
            raise ValueError(
                "phase3 backend_scheduled raw oracle execution currently requires phase3_oracle_raw_transport='auto'."
            )
    else:
        if noise_mode != "runtime":
            raise ValueError(
                "phase3 raw oracle execution currently supports only noise_mode in {'runtime','backend_scheduled'}."
            )
        if bool(config.use_fake_backend):
            raise ValueError(
                "phase3 raw oracle execution requires a real runtime backend when noise_mode='runtime'."
            )
        if raw_transport not in {"auto", "sampler_v2"}:
            raise ValueError(
                "phase3_oracle_raw_transport must be one of {'auto','sampler_v2'}."
            )
    if int(config.transpile_optimization_level) not in {0, 1, 2, 3}:
        raise ValueError(
            "phase3_oracle_transpile_optimization_level must be one of {0,1,2,3}."
        )


def _oracle_mitigation_payload_from_fields(
    *,
    mitigation_mode: str,
    local_readout_strategy: str | None,
    zne_scales: Sequence[float] = (),
    dd_sequence: str | None = None,
    local_gate_twirling: bool = False,
) -> dict[str, Any]:
    mitigation_mode_key = str(mitigation_mode).strip().lower()
    local_readout_strategy_key = (
        "mthree"
        if mitigation_mode_key == "readout" and local_readout_strategy in {None, ""}
        else (
            None
            if local_readout_strategy in {None, ""}
            else str(local_readout_strategy).strip().lower()
        )
    )
    payload = {
        "mode": str(mitigation_mode_key),
        "zne_scales": [float(x) for x in zne_scales],
        "dd_sequence": (
            None
            if dd_sequence in {None, "", "none"}
            else str(dd_sequence).strip()
        ),
        "local_readout_strategy": local_readout_strategy_key,
    }
    if bool(local_gate_twirling):
        payload["local_gate_twirling"] = True
    return payload


def _phase3_oracle_mitigation_payload(config: Phase3OracleGradientConfig) -> dict[str, Any]:
    return _oracle_mitigation_payload_from_fields(
        mitigation_mode=str(config.mitigation_mode),
        local_readout_strategy=config.local_readout_strategy,
        zne_scales=tuple(getattr(config, "zne_scales", ()) or ()),
        dd_sequence=getattr(config, "dd_sequence", None),
        local_gate_twirling=bool(getattr(config, "local_gate_twirling", False)),
    )


def _validate_final_noise_audit_config(
    *,
    config: FinalNoiseAuditConfig,
    problem: str,
) -> None:
    config = _resolve_final_noise_audit_config(config)
    problem_key = str(problem).strip().lower()
    if problem_key != "hh":
        raise ValueError("final noise audit is currently only valid for problem='hh'.")
    if str(config.noise_mode) not in {"ideal", "shots", "aer_noise", "backend_scheduled", "runtime"}:
        raise ValueError(
            "final_noise_audit_mode must be one of {'off','ideal','shots','aer_noise','backend_scheduled','runtime'}."
        )
    if int(config.shots) < 1:
        raise ValueError("final_noise_audit_shots must be >= 1.")
    if int(config.oracle_repeats) < 1:
        raise ValueError("final_noise_audit_repeats must be >= 1.")
    if str(config.oracle_aggregate) != "mean":
        raise ValueError("final noise audit currently requires oracle_aggregate='mean'.")
    if int(config.transpile_optimization_level) not in {0, 1, 2, 3}:
        raise ValueError(
            "final_noise_audit_transpile_optimization_level must be one of {0,1,2,3}."
        )
    mitigation_mode = str(config.mitigation_mode).strip().lower()
    if mitigation_mode not in {"none", "readout"}:
        raise ValueError("final_noise_audit_mitigation must be one of {'none','readout'}.")
    zne_scales = tuple(float(x) for x in getattr(config, "zne_scales", ()) or ())
    local_gate_twirling = bool(getattr(config, "local_gate_twirling", False))
    dd_sequence = (
        None
        if getattr(config, "dd_sequence", None) in {None, "", "none"}
        else str(getattr(config, "dd_sequence")).strip()
    )
    runtime_profile_name = str(config.runtime_profile_name)
    runtime_session_policy = str(config.runtime_session_policy)
    if runtime_profile_name not in _FINAL_NOISE_AUDIT_RUNTIME_PROFILE_NAMES:
        raise ValueError(
            "final_noise_audit_runtime_profile must be one of "
            f"{sorted(_FINAL_NOISE_AUDIT_RUNTIME_PROFILE_NAMES)}."
        )
    if runtime_session_policy not in _FINAL_NOISE_AUDIT_RUNTIME_SESSION_POLICIES:
        raise ValueError(
            "final_noise_audit_runtime_session_policy must be one of "
            f"{sorted(_FINAL_NOISE_AUDIT_RUNTIME_SESSION_POLICIES)}."
        )
    local_readout_strategy = (
        None
        if config.local_readout_strategy in {None, ""}
        else str(config.local_readout_strategy)
    )
    noise_mode = str(config.noise_mode)
    if mitigation_mode == "readout":
        if noise_mode == "backend_scheduled":
            if local_readout_strategy not in {None, "mthree"}:
                raise ValueError(
                    "final_noise_audit_local_readout_strategy must be 'mthree' when backend_scheduled readout mitigation is enabled."
                )
        elif noise_mode == "runtime":
            if local_readout_strategy is not None:
                raise ValueError(
                    "final noise audit runtime readout uses provider-side mitigation and does not accept local readout strategy."
                )
        else:
            raise ValueError(
                "final noise audit readout mitigation is currently supported only for noise_mode in {'backend_scheduled','runtime'}."
            )
    elif local_readout_strategy is not None:
        raise ValueError(
            "final_noise_audit_local_readout_strategy requires final_noise_audit_mitigation='readout'."
        )
    if noise_mode != "backend_scheduled" and (
        zne_scales
        or local_gate_twirling
        or dd_sequence not in {None, "", "none"}
    ):
        raise ValueError(
            "final noise audit local ZNE/gate twirling/DD currently require noise_mode='backend_scheduled'."
        )
    if noise_mode == "backend_scheduled" and zne_scales:
        _validate_backend_scheduled_local_zne_scales(
            zne_scales,
            field_name="final_noise_audit_zne_scales",
        )
    if noise_mode == "backend_scheduled":
        if runtime_profile_name != "legacy_runtime_v0":
            raise ValueError(
                "final_noise_audit_runtime_profile is only valid for final_noise_audit_mode='runtime'."
            )
        if runtime_session_policy != "prefer_session":
            raise ValueError(
                "final_noise_audit_runtime_session_policy is only valid for final_noise_audit_mode='runtime'."
            )
        if not bool(config.use_fake_backend):
            raise ValueError(
                "final noise audit backend_scheduled mode requires --final-noise-audit-use-fake-backend."
            )
        if config.backend_name in {None, ""}:
            raise ValueError(
                "final noise audit backend_scheduled mode requires --final-noise-audit-backend-name."
            )
    elif noise_mode == "runtime":
        if config.backend_name in {None, ""}:
            raise ValueError(
                "final noise audit runtime mode requires --final-noise-audit-backend-name."
            )
        if bool(config.use_fake_backend):
            raise ValueError(
                "final noise audit runtime mode requires a real runtime backend; do not enable --final-noise-audit-use-fake-backend."
            )
        if runtime_profile_name != "legacy_runtime_v0" and mitigation_mode != "none":
            raise ValueError(
                "final noise audit runtime profiles already encode mitigation/suppression; use final_noise_audit_mitigation='none' when final_noise_audit_runtime_profile is explicit."
            )
        if zne_scales or local_gate_twirling or dd_sequence not in {None, "", "none"}:
            raise ValueError(
                "final noise audit runtime full suppression stacks should use an explicit runtime profile, not local backend_scheduled knobs."
            )
    else:
        if runtime_profile_name != "legacy_runtime_v0":
            raise ValueError(
                "final_noise_audit_runtime_profile is only valid for final_noise_audit_mode='runtime'."
            )
        if runtime_session_policy != "prefer_session":
            raise ValueError(
                "final_noise_audit_runtime_session_policy is only valid for final_noise_audit_mode='runtime'."
            )


def _final_noise_audit_config_payload(
    config: FinalNoiseAuditConfig | None,
) -> dict[str, Any] | None:
    if config is None:
        return None
    config = _resolve_final_noise_audit_config(config)
    return {
        "noise_mode": str(config.noise_mode),
        "shots": int(config.shots),
        "oracle_repeats": int(config.oracle_repeats),
        "oracle_aggregate": str(config.oracle_aggregate),
        "backend_name": (None if config.backend_name in {None, ""} else str(config.backend_name)),
        "use_fake_backend": bool(config.use_fake_backend),
        "seed": int(config.seed),
        "mitigation": dict(
            _oracle_mitigation_payload_from_fields(
                mitigation_mode=str(config.mitigation_mode),
                local_readout_strategy=config.local_readout_strategy,
                zne_scales=tuple(getattr(config, "zne_scales", ()) or ()),
                dd_sequence=getattr(config, "dd_sequence", None),
                local_gate_twirling=bool(getattr(config, "local_gate_twirling", False)),
            )
        ),
        "runtime_profile": {"name": str(config.runtime_profile_name)},
        "runtime_session": {"mode": str(config.runtime_session_policy)},
        "compare_unmitigated_baseline": bool(config.compare_unmitigated_baseline),
        "execution_surface": "expectation_v1",
        "seed_transpiler": config.seed_transpiler,
        "transpile_optimization_level": int(config.transpile_optimization_level),
        "strict": bool(config.strict),
    }


def _phase3_oracle_gradient_config_payload(
    config: Phase3OracleGradientConfig | None,
) -> dict[str, Any] | None:
    if config is None:
        return None
    config = _resolve_phase3_oracle_gradient_config(config)
    return {
        "noise_mode": str(config.noise_mode),
        "shots": int(config.shots),
        "oracle_repeats": int(config.oracle_repeats),
        "oracle_aggregate": str(config.oracle_aggregate),
        "backend_name": (None if config.backend_name in {None, ""} else str(config.backend_name)),
        "use_fake_backend": bool(config.use_fake_backend),
        "seed": int(config.seed),
        "gradient_step": float(config.gradient_step),
        "mitigation": dict(_phase3_oracle_mitigation_payload(config)),
        "scope": str(config.scope),
        "execution_surface_requested": str(config.execution_surface_requested),
        "execution_surface": str(config.execution_surface),
        "raw_transport": str(config.raw_transport),
        "raw_store_memory": bool(config.raw_store_memory),
        "raw_artifact_path": config.raw_artifact_path,
        "seed_transpiler": config.seed_transpiler,
        "transpile_optimization_level": int(config.transpile_optimization_level),
    }


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


@dataclass(frozen=True)
class ResolvedBeamCapacityPolicy:
    live_branches_requested: int
    children_per_parent_requested: int | None
    terminated_keep_requested: int | None
    live_branches_effective: int
    children_per_parent_effective: int
    terminated_keep_effective: int
    beam_enabled: bool
    source_children_per_parent: str
    source_terminated_keep: str


def _resolve_beam_capacity_policy(
    *,
    adapt_beam_live_branches: int,
    adapt_beam_children_per_parent: int | None,
    adapt_beam_terminated_keep: int | None,
) -> ResolvedBeamCapacityPolicy:
    live_requested = int(adapt_beam_live_branches)
    children_requested = (
        None if adapt_beam_children_per_parent is None else int(adapt_beam_children_per_parent)
    )
    terminated_requested = (
        None if adapt_beam_terminated_keep is None else int(adapt_beam_terminated_keep)
    )
    if live_requested < 1:
        raise ValueError("adapt_beam_live_branches must be >= 1.")
    if children_requested is not None and children_requested < 1:
        raise ValueError("adapt_beam_children_per_parent must be >= 1 when provided.")
    if terminated_requested is not None and terminated_requested < 1:
        raise ValueError("adapt_beam_terminated_keep must be >= 1 when provided.")
    if live_requested == 1:
        return ResolvedBeamCapacityPolicy(
            live_branches_requested=live_requested,
            children_per_parent_requested=children_requested,
            terminated_keep_requested=terminated_requested,
            live_branches_effective=1,
            children_per_parent_effective=1,
            terminated_keep_effective=1,
            beam_enabled=False,
            source_children_per_parent=(
                "single_branch_clamp.explicit" if children_requested is not None else "single_branch_clamp.default"
            ),
            source_terminated_keep=(
                "single_branch_clamp.explicit" if terminated_requested is not None else "single_branch_clamp.default"
            ),
        )
    children_resolved_raw = 2 if children_requested is None else int(children_requested)
    children_effective = min(int(children_resolved_raw), int(live_requested))
    terminated_effective = (
        int(live_requested) if terminated_requested is None else int(terminated_requested)
    )
    return ResolvedBeamCapacityPolicy(
        live_branches_requested=live_requested,
        children_per_parent_requested=children_requested,
        terminated_keep_requested=terminated_requested,
        live_branches_effective=int(live_requested),
        children_per_parent_effective=int(children_effective),
        terminated_keep_effective=int(terminated_effective),
        beam_enabled=True,
        source_children_per_parent=(
            "default_live_gt_1_cap2" if children_requested is None else "explicit"
        ),
        source_terminated_keep=(
            "default_live_width" if terminated_requested is None else "explicit"
        ),
    )


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


def _estimate_stderr_value(estimate: Any) -> float:
    if isinstance(estimate, Mapping):
        raw_value = estimate.get("stderr")
    else:
        raw_value = getattr(estimate, "stderr", None)
    if raw_value is None:
        raise ValueError("Oracle estimate must expose a finite nonnegative stderr.")
    stderr_value = float(raw_value)
    if (not math.isfinite(stderr_value)) or stderr_value < 0.0:
        raise ValueError("Oracle estimate stderr must be finite and nonnegative.")
    return float(stderr_value)


def _oracle_fd_gradient_stderr(
    e_plus: Any,
    e_minus: Any,
    *,
    grad_step: float,
) -> float:
    step = float(grad_step)
    if (not math.isfinite(step)) or step <= 0.0:
        raise ValueError("grad_step must be finite and > 0 for oracle finite-difference stderr.")
    stderr_plus = _estimate_stderr_value(e_plus)
    stderr_minus = _estimate_stderr_value(e_minus)
    grad_stderr = math.sqrt(stderr_plus ** 2 + stderr_minus ** 2) / (2.0 * step)
    if (not math.isfinite(grad_stderr)) or grad_stderr < 0.0:
        raise ValueError("Resolved oracle finite-difference stderr must be finite and nonnegative.")
    return float(grad_stderr)


def _phase3_sigma_hat_for_label(
    *,
    candidate_label: str,
    sigma_by_label: Mapping[str, float],
    phase3_enabled: bool,
) -> float:
    if not bool(phase3_enabled):
        return 0.0
    sigma_raw = sigma_by_label.get(str(candidate_label))
    if sigma_raw is None:
        return 0.0
    sigma_value = float(sigma_raw)
    if (not math.isfinite(sigma_value)) or sigma_value < 0.0:
        return 0.0
    return float(sigma_value)


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
    current_layout = build_parameter_layout(
        ops,
        ignore_identity=True,
        coefficient_tolerance=1e-12,
        sort_terms=True,
    )
    op_layout = build_parameter_layout(
        [op],
        ignore_identity=True,
        coefficient_tolerance=1e-12,
        sort_terms=True,
    )
    pos_logical = max(0, min(int(current_layout.logical_parameter_count), int(position_id)))
    pos_runtime = int(runtime_insert_position(current_layout, pos_logical))
    new_ops = list(ops)
    new_ops.insert(pos_logical, op)
    theta_arr = np.asarray(theta, dtype=float).reshape(-1)
    insert_block = np.full(int(op_layout.runtime_parameter_count), float(init_theta), dtype=float)
    new_theta = np.insert(theta_arr, pos_runtime, insert_block)
    return new_ops, np.asarray(new_theta, dtype=float)


def _splice_logical_candidate_at_position(
    *,
    ops: list[AnsatzTerm],
    theta: np.ndarray,
    candidate: _ADAPTLogicalCandidate,
    pool: Sequence[AnsatzTerm],
    position_id: int,
    init_theta_values: Sequence[float] | None = None,
) -> tuple[list[AnsatzTerm], np.ndarray]:
    theta_arr = np.asarray(theta, dtype=float).reshape(-1)
    append_position = int(theta_arr.size)
    pos = max(0, min(int(append_position), int(position_id)))
    theta_values = (
        [0.0] * int(len(candidate.pool_indices))
        if init_theta_values is None
        else [float(x) for x in init_theta_values]
    )
    if len(theta_values) != int(len(candidate.pool_indices)):
        raise ValueError("Logical candidate insertion requires one init theta per emitted term.")

    new_ops = list(ops)
    new_theta = np.asarray(theta_arr, dtype=float)
    for offset, (pool_idx, theta_val) in enumerate(zip(candidate.pool_indices, theta_values)):
        insert_at = int(pos + offset)
        new_ops.insert(insert_at, pool[int(pool_idx)])
        new_theta = np.insert(new_theta, insert_at, float(theta_val))
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
    full_score = record.get("full_v2_score", float("-inf"))
    if full_score is None:
        full_score = float("-inf")
    cheap_score = record.get("cheap_score", record.get("simple_score", float("-inf")))
    if cheap_score is None:
        cheap_score = record.get("simple_score", float("-inf"))
    return (
        -float(full_score),
        -float(cheap_score),
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
    adapt_analytic_noise_std: float = 0.0,
    adapt_analytic_noise_seed: int | None = None,
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
    adapt_ref_json: Path | None = None,
    adapt_gradient_parity_check: bool = False,
    adapt_state_backend: str = "compiled",
    adapt_reopt_policy: str = "append_only",
    adapt_window_size: int = 3,
    adapt_window_topk: int = 0,
    adapt_full_refit_every: int = 0,
    adapt_final_full_refit: bool = True,
    exact_gs_override: float | None = None,
    adapt_continuation_mode: str | None = "phase3_v1",
    phase1_lambda_F: float = 1.0,
    phase1_lambda_compile: float = 0.05,
    phase1_lambda_measure: float = 0.02,
    phase1_lambda_leak: float = 0.0,
    phase1_score_z_alpha: float = 0.0,
    phase1_depth_ref: float = 1.0,
    phase1_group_ref: float = 1.0,
    phase1_shot_ref: float = 1.0,
    phase1_family_ref: float = 1.0,
    phase1_compile_cx_proxy_weight: float = 1.0,
    phase1_compile_sq_proxy_weight: float = 0.5,
    phase1_compile_rotation_step_weight: float = 1.0,
    phase1_compile_position_shift_weight: float = 1.0,
    phase1_compile_refit_active_weight: float = 1.0,
    phase1_measure_groups_weight: float = 1.0,
    phase1_measure_shots_weight: float = 1.0,
    phase1_measure_reuse_weight: float = 1.0,
    phase1_opt_dim_cost_scale: float = 1.0,
    phase1_family_repeat_cost_scale: float = 1.0,
    phase1_shortlist_size: int = 64,
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
    phase2_score_z_alpha: float | None = None,
    phase2_lambda_F: float | None = None,
    phase2_depth_ref: float = 1.0,
    phase2_group_ref: float = 1.0,
    phase2_shot_ref: float = 1.0,
    phase2_optdim_ref: float = 1.0,
    phase2_reuse_ref: float = 1.0,
    phase2_family_ref: float = 1.0,
    phase2_novelty_eps: float = 1e-6,
    phase2_cheap_score_eps: float = 1e-12,
    phase2_metric_floor: float = 1e-12,
    phase2_reduced_metric_collapse_rel_tol: float = 1e-8,
    phase2_ridge_growth_factor: float = 10.0,
    phase2_ridge_max_steps: int = 12,
    phase2_leakage_cap: float = 1e6,
    phase2_compile_cx_proxy_weight: float = 1.0,
    phase2_compile_sq_proxy_weight: float = 0.5,
    phase2_compile_rotation_step_weight: float = 1.0,
    phase2_compile_position_shift_weight: float = 1.0,
    phase2_compile_refit_active_weight: float = 1.0,
    phase2_measure_groups_weight: float = 1.0,
    phase2_measure_shots_weight: float = 1.0,
    phase2_measure_reuse_weight: float = 1.0,
    phase2_opt_dim_cost_scale: float = 1.0,
    phase2_family_repeat_cost_scale: float = 1.0,
    phase2_w_depth: float = 0.2,
    phase2_w_group: float = 0.15,
    phase2_w_shot: float = 0.15,
    phase2_w_optdim: float = 0.1,
    phase2_w_reuse: float = 0.1,
    phase2_w_lifetime: float = 0.05,
    phase2_eta_L: float = 0.0,
    phase2_motif_bonus_weight: float = 0.05,
    phase2_duplicate_penalty_weight: float = 0.0,
    phase2_frontier_ratio: float = 0.9,
    phase3_frontier_ratio: float = 0.9,
    phase3_tie_beam_score_ratio: float = 1.0,
    phase3_tie_beam_abs_tol: float = 0.0,
    phase3_tie_beam_max_branches: int = 1,
    phase3_tie_beam_max_late_coordinate: float = 1.0,
    phase3_tie_beam_min_depth_left: int = 0,
    phase2_enable_batching: bool = True,
    phase2_batch_target_size: int = 2,
    phase2_batch_size_cap: int = 3,
    phase2_batch_near_degenerate_ratio: float = 0.9,
    phase2_batch_rank_rel_tol: float = 1e-6,
    phase2_batch_additivity_tol: float = 0.25,
    phase2_compat_overlap_weight: float = 0.4,
    phase2_compat_comm_weight: float = 0.2,
    phase2_compat_curv_weight: float = 0.2,
    phase2_compat_sched_weight: float = 0.2,
    phase2_compat_measure_weight: float = 0.2,
    phase2_remaining_evaluations_proxy_mode: str = "auto",
    adapt_pool_class_filter_json: Path | None = None,
    phase3_motif_source_json: Path | None = None,
    phase3_symmetry_mitigation_mode: str = "off",
    phase3_enable_rescue: bool = False,
    phase3_lifetime_cost_mode: str = "phase3_v1",
    phase3_runtime_split_mode: str = "off",
    phase3_backend_cost_mode: str = "proxy",
    phase3_backend_name: str | None = None,
    phase3_backend_shortlist: Sequence[str] | None = None,
    phase3_backend_transpile_seed: int = 7,
    phase3_backend_optimization_level: int = 1,
    phase3_oracle_gradient_config: Phase3OracleGradientConfig | None = None,
    final_noise_audit_config: FinalNoiseAuditConfig | None = None,
    phase3_oracle_inner_objective_mode: str = "exact",
    phase3_selector_debug_topk: int = 0,
    phase3_selector_debug_max_depth: int = 0,
    adapt_beam_live_branches: int = 1,
    adapt_beam_children_per_parent: int | None = None,
    adapt_beam_terminated_keep: int | None = None,
    diagnostics_out: dict[str, Any] | None = None,
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
    phase1_shortlist_size_val = int(phase1_shortlist_size)
    phase3_tie_beam_score_ratio_val = float(phase3_tie_beam_score_ratio)
    if (not math.isfinite(phase3_tie_beam_score_ratio_val)) or phase3_tie_beam_score_ratio_val <= 0.0:
        raise ValueError("phase3_tie_beam_score_ratio must be finite and > 0.")
    phase3_tie_beam_abs_tol_val = float(phase3_tie_beam_abs_tol)
    if (not math.isfinite(phase3_tie_beam_abs_tol_val)) or phase3_tie_beam_abs_tol_val < 0.0:
        raise ValueError("phase3_tie_beam_abs_tol must be finite and >= 0.")
    phase3_tie_beam_max_branches_val = int(phase3_tie_beam_max_branches)
    if phase3_tie_beam_max_branches_val < 1:
        raise ValueError("phase3_tie_beam_max_branches must be >= 1.")
    phase3_tie_beam_max_late_coordinate_val = float(phase3_tie_beam_max_late_coordinate)
    if not math.isfinite(phase3_tie_beam_max_late_coordinate_val):
        raise ValueError("phase3_tie_beam_max_late_coordinate must be finite.")
    phase3_tie_beam_max_late_coordinate_val = float(
        max(0.0, min(1.0, phase3_tie_beam_max_late_coordinate_val))
    )
    phase3_tie_beam_min_depth_left_val = int(phase3_tie_beam_min_depth_left)
    if phase3_tie_beam_min_depth_left_val < 0:
        raise ValueError("phase3_tie_beam_min_depth_left must be >= 0.")
    if adapt_window_size_val < 1:
        raise ValueError("adapt_window_size must be >= 1.")
    if adapt_window_topk_val < 0:
        raise ValueError("adapt_window_topk must be >= 0.")
    if adapt_full_refit_every_val < 0:
        raise ValueError("adapt_full_refit_every must be >= 0.")
    if phase1_shortlist_size_val < 1:
        raise ValueError("phase1_shortlist_size must be >= 1.")
    phase1_depth_ref_val = float(phase1_depth_ref)
    phase1_group_ref_val = float(phase1_group_ref)
    phase1_shot_ref_val = float(phase1_shot_ref)
    phase1_family_ref_val = float(phase1_family_ref)
    for ref_name, ref_val in (
        ("phase1_depth_ref", phase1_depth_ref_val),
        ("phase1_group_ref", phase1_group_ref_val),
        ("phase1_shot_ref", phase1_shot_ref_val),
        ("phase1_family_ref", phase1_family_ref_val),
    ):
        if (not math.isfinite(ref_val)) or ref_val <= 0.0:
            raise ValueError(f"{ref_name} must be finite and > 0.")
    phase1_cost_scale_vals = (
        ("phase1_compile_cx_proxy_weight", float(phase1_compile_cx_proxy_weight)),
        ("phase1_compile_sq_proxy_weight", float(phase1_compile_sq_proxy_weight)),
        ("phase1_compile_rotation_step_weight", float(phase1_compile_rotation_step_weight)),
        ("phase1_compile_position_shift_weight", float(phase1_compile_position_shift_weight)),
        ("phase1_compile_refit_active_weight", float(phase1_compile_refit_active_weight)),
        ("phase1_measure_groups_weight", float(phase1_measure_groups_weight)),
        ("phase1_measure_shots_weight", float(phase1_measure_shots_weight)),
        ("phase1_measure_reuse_weight", float(phase1_measure_reuse_weight)),
        ("phase1_opt_dim_cost_scale", float(phase1_opt_dim_cost_scale)),
        ("phase1_family_repeat_cost_scale", float(phase1_family_repeat_cost_scale)),
    )
    for coeff_name, coeff_val in phase1_cost_scale_vals:
        if (not math.isfinite(coeff_val)) or coeff_val < 0.0:
            raise ValueError(f"{coeff_name} must be finite and >= 0.")
    phase2_score_z_alpha_val = float(
        phase1_score_z_alpha if phase2_score_z_alpha is None else phase2_score_z_alpha
    )
    if (not math.isfinite(phase2_score_z_alpha_val)) or phase2_score_z_alpha_val < 0.0:
        raise ValueError("phase2_score_z_alpha must be finite and >= 0.")
    phase2_lambda_F_val = float(phase1_lambda_F if phase2_lambda_F is None else phase2_lambda_F)
    if (not math.isfinite(phase2_lambda_F_val)) or phase2_lambda_F_val <= 0.0:
        raise ValueError("phase2_lambda_F must be finite and > 0.")
    phase2_depth_ref_val = float(phase2_depth_ref)
    phase2_group_ref_val = float(phase2_group_ref)
    phase2_shot_ref_val = float(phase2_shot_ref)
    phase2_optdim_ref_val = float(phase2_optdim_ref)
    phase2_reuse_ref_val = float(phase2_reuse_ref)
    phase2_family_ref_val = float(phase2_family_ref)
    for ref_name, ref_val in (
        ("phase2_depth_ref", phase2_depth_ref_val),
        ("phase2_group_ref", phase2_group_ref_val),
        ("phase2_shot_ref", phase2_shot_ref_val),
        ("phase2_optdim_ref", phase2_optdim_ref_val),
        ("phase2_reuse_ref", phase2_reuse_ref_val),
        ("phase2_family_ref", phase2_family_ref_val),
    ):
        if (not math.isfinite(ref_val)) or ref_val <= 0.0:
            raise ValueError(f"{ref_name} must be finite and > 0.")
    phase2_novelty_eps_val = float(phase2_novelty_eps)
    if (not math.isfinite(phase2_novelty_eps_val)) or phase2_novelty_eps_val < 0.0:
        raise ValueError("phase2_novelty_eps must be finite and >= 0.")
    phase2_cheap_score_eps_val = float(phase2_cheap_score_eps)
    if (not math.isfinite(phase2_cheap_score_eps_val)) or phase2_cheap_score_eps_val <= 0.0:
        raise ValueError("phase2_cheap_score_eps must be finite and > 0.")
    phase2_metric_floor_val = float(phase2_metric_floor)
    if (not math.isfinite(phase2_metric_floor_val)) or phase2_metric_floor_val <= 0.0:
        raise ValueError("phase2_metric_floor must be finite and > 0.")
    phase2_reduced_metric_collapse_rel_tol_val = float(phase2_reduced_metric_collapse_rel_tol)
    if (
        (not math.isfinite(phase2_reduced_metric_collapse_rel_tol_val))
        or phase2_reduced_metric_collapse_rel_tol_val < 0.0
    ):
        raise ValueError(
            "phase2_reduced_metric_collapse_rel_tol must be finite and >= 0."
        )
    phase2_ridge_growth_factor_val = float(phase2_ridge_growth_factor)
    if (not math.isfinite(phase2_ridge_growth_factor_val)) or phase2_ridge_growth_factor_val <= 1.0:
        raise ValueError("phase2_ridge_growth_factor must be finite and > 1.")
    phase2_ridge_max_steps_val = int(phase2_ridge_max_steps)
    if phase2_ridge_max_steps_val < 1:
        raise ValueError("phase2_ridge_max_steps must be >= 1.")
    phase2_leakage_cap_val = float(phase2_leakage_cap)
    if (not math.isfinite(phase2_leakage_cap_val)) or phase2_leakage_cap_val <= 0.0:
        raise ValueError("phase2_leakage_cap must be finite and > 0.")
    phase2_cost_scale_vals = (
        ("phase2_compile_cx_proxy_weight", float(phase2_compile_cx_proxy_weight)),
        ("phase2_compile_sq_proxy_weight", float(phase2_compile_sq_proxy_weight)),
        ("phase2_compile_rotation_step_weight", float(phase2_compile_rotation_step_weight)),
        ("phase2_compile_position_shift_weight", float(phase2_compile_position_shift_weight)),
        ("phase2_compile_refit_active_weight", float(phase2_compile_refit_active_weight)),
        ("phase2_measure_groups_weight", float(phase2_measure_groups_weight)),
        ("phase2_measure_shots_weight", float(phase2_measure_shots_weight)),
        ("phase2_measure_reuse_weight", float(phase2_measure_reuse_weight)),
        ("phase2_opt_dim_cost_scale", float(phase2_opt_dim_cost_scale)),
        ("phase2_family_repeat_cost_scale", float(phase2_family_repeat_cost_scale)),
    )
    for coeff_name, coeff_val in phase2_cost_scale_vals:
        if (not math.isfinite(coeff_val)) or coeff_val < 0.0:
            raise ValueError(f"{coeff_name} must be finite and >= 0.")
    phase3_selector_debug_topk_val = max(0, int(phase3_selector_debug_topk))
    phase3_selector_debug_max_depth_val = max(0, int(phase3_selector_debug_max_depth))
    adapt_inner_optimizer_key = str(adapt_inner_optimizer).strip().upper()
    if adapt_inner_optimizer_key not in {"COBYLA", "POWELL", "SPSA"}:
        raise ValueError("adapt_inner_optimizer must be one of {'COBYLA','POWELL','SPSA'}.")
    adapt_spsa_eval_agg_key = str(adapt_spsa_eval_agg).strip().lower()
    if adapt_spsa_eval_agg_key not in {"mean", "median"}:
        raise ValueError("adapt_spsa_eval_agg must be one of {'mean','median'}.")
    if int(adapt_spsa_callback_every) < 1:
        raise ValueError("adapt_spsa_callback_every must be >= 1.")
    if float(adapt_spsa_progress_every_s) < 0.0:
        raise ValueError("adapt_spsa_progress_every_s must be >= 0.")
    adapt_analytic_noise_std_val = float(adapt_analytic_noise_std)
    if (not math.isfinite(adapt_analytic_noise_std_val)) or adapt_analytic_noise_std_val < 0.0:
        raise ValueError("adapt_analytic_noise_std must be finite and >= 0.")
    adapt_analytic_noise_seed_val = (
        None if adapt_analytic_noise_seed is None else int(adapt_analytic_noise_seed)
    )
    adapt_analytic_noise_enabled = bool(adapt_analytic_noise_std_val > 0.0)
    adapt_noise_rng = np.random.default_rng(adapt_analytic_noise_seed_val)
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
    if phase3_oracle_gradient_config is not None:
        phase3_oracle_gradient_config = _resolve_phase3_oracle_gradient_config(
            phase3_oracle_gradient_config
        )
        _validate_phase3_oracle_gradient_config(
            config=phase3_oracle_gradient_config,
            problem=str(problem_key),
            continuation_mode=str(continuation_mode),
        )
    if final_noise_audit_config is not None:
        final_noise_audit_config = _resolve_final_noise_audit_config(final_noise_audit_config)
        _validate_final_noise_audit_config(
            config=final_noise_audit_config,
            problem=str(problem_key),
        )
    phase3_oracle_inner_objective_mode_requested_key = (
        str(phase3_oracle_inner_objective_mode).strip().lower() or "exact"
    )
    if phase3_oracle_inner_objective_mode_requested_key not in {"exact", "noisy_v1"}:
        raise ValueError(
            "phase3_oracle_inner_objective_mode must be one of {'exact','noisy_v1'}."
        )
    phase3_oracle_inner_objective_requested = bool(
        phase3_oracle_inner_objective_mode_requested_key == "noisy_v1"
    )
    phase3_oracle_inner_objective_mode_key = str(
        phase3_oracle_inner_objective_mode_requested_key
    )
    phase3_oracle_inner_objective_runtime_guard_reason: str | None = None
    phase3_oracle_inner_objective_enabled = bool(
        phase3_oracle_inner_objective_requested
    )
    if phase3_oracle_inner_objective_enabled:
        if phase3_oracle_gradient_config is None:
            raise ValueError(
                "phase3_oracle_inner_objective_mode='noisy_v1' requires an active phase3 oracle gradient config."
            )
        if adapt_inner_optimizer_key != "SPSA":
            raise ValueError(
                "phase3_oracle_inner_objective_mode='noisy_v1' currently requires adapt_inner_optimizer='SPSA'."
            )
        oracle_inner_execution_surface = str(
            phase3_oracle_gradient_config.execution_surface
        ).strip().lower()
        if oracle_inner_execution_surface not in {
            "expectation_v1",
            "raw_measurement_v1",
        }:
            raise ValueError(
                "phase3_oracle_inner_objective_mode='noisy_v1' currently requires phase3_oracle_execution_surface in {'expectation_v1','raw_measurement_v1'}."
            )
    phase3_oracle_inner_backend_name = "exact_statevector"
    if phase3_oracle_inner_objective_enabled and phase3_oracle_gradient_config is not None:
        phase3_oracle_inner_backend_name = (
            "oracle_raw_measurement_v1"
            if str(phase3_oracle_gradient_config.execution_surface).strip().lower()
            == "raw_measurement_v1"
            else "oracle_expectation_v1"
        )
    stop_policy = _resolve_adapt_stop_policy(
        problem=str(problem_key),
        continuation_mode=str(continuation_mode),
        adapt_drop_floor=adapt_drop_floor,
        adapt_drop_patience=adapt_drop_patience,
        adapt_drop_min_depth=adapt_drop_min_depth,
        adapt_grad_floor=adapt_grad_floor,
    )
    beam_policy = _resolve_beam_capacity_policy(
        adapt_beam_live_branches=int(adapt_beam_live_branches),
        adapt_beam_children_per_parent=adapt_beam_children_per_parent,
        adapt_beam_terminated_keep=adapt_beam_terminated_keep,
    )
    phase3_tie_beam_enabled = bool(
        str(problem_key) == "hh"
        and str(continuation_mode).strip().lower() == "phase3_v1"
        and int(phase3_tie_beam_max_branches_val) > 1
        and (
            float(phase3_tie_beam_score_ratio_val) < 1.0
            or float(phase3_tie_beam_abs_tol_val) > 0.0
        )
    )
    if phase3_tie_beam_enabled and not bool(beam_policy.beam_enabled):
        beam_policy = ResolvedBeamCapacityPolicy(
            live_branches_requested=int(beam_policy.live_branches_requested),
            children_per_parent_requested=beam_policy.children_per_parent_requested,
            terminated_keep_requested=beam_policy.terminated_keep_requested,
            live_branches_effective=1,
            children_per_parent_effective=1,
            terminated_keep_effective=int(max(1, int(phase3_tie_beam_max_branches_val))),
            beam_enabled=True,
            source_children_per_parent="conditional_phase3_tie_band_base1",
            source_terminated_keep="conditional_phase3_tie_band_base1",
        )
    if bool(beam_policy.beam_enabled) and not (
        str(problem_key) == "hh"
        and str(continuation_mode).strip().lower() in _HH_STAGED_CONTINUATION_MODES
    ):
        raise ValueError(
            "True ADAPT beam mode is currently supported only for HH staged continuation modes."
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
    phase3_tie_beam_score_ratio_val = float(phase3_tie_beam_score_ratio)
    if (not math.isfinite(phase3_tie_beam_score_ratio_val)) or phase3_tie_beam_score_ratio_val <= 0.0:
        raise ValueError("phase3_tie_beam_score_ratio must be finite and > 0.")
    phase3_tie_beam_abs_tol_val = float(phase3_tie_beam_abs_tol)
    if (not math.isfinite(phase3_tie_beam_abs_tol_val)) or phase3_tie_beam_abs_tol_val < 0.0:
        raise ValueError("phase3_tie_beam_abs_tol must be finite and >= 0.")
    phase3_tie_beam_max_branches_val = int(phase3_tie_beam_max_branches)
    if phase3_tie_beam_max_branches_val < 1:
        raise ValueError("phase3_tie_beam_max_branches must be >= 1.")
    phase3_tie_beam_max_late_coordinate_val = float(phase3_tie_beam_max_late_coordinate)
    if not math.isfinite(phase3_tie_beam_max_late_coordinate_val):
        raise ValueError("phase3_tie_beam_max_late_coordinate must be finite.")
    phase3_tie_beam_max_late_coordinate_val = float(
        max(0.0, min(1.0, phase3_tie_beam_max_late_coordinate_val))
    )
    phase3_tie_beam_min_depth_left_val = int(phase3_tie_beam_min_depth_left)
    if phase3_tie_beam_min_depth_left_val < 0:
        raise ValueError("phase3_tie_beam_min_depth_left must be >= 0.")
    phase3_backend_cost_mode_key = str(phase3_backend_cost_mode).strip().lower()
    if phase3_backend_cost_mode_key not in {"proxy", "transpile_single_v1", "transpile_shortlist_v1"}:
        raise ValueError(
            "phase3_backend_cost_mode must be one of {'proxy','transpile_single_v1','transpile_shortlist_v1'}."
        )
    phase3_backend_shortlist_tokens = tuple(
        str(tok).strip()
        for tok in (str(phase3_backend_shortlist).split(",") if isinstance(phase3_backend_shortlist, str) else list(phase3_backend_shortlist or []))
        if str(tok).strip() != ""
    )
    if phase3_backend_cost_mode_key != "proxy":
        if str(problem_key) != "hh":
            raise ValueError("phase3_backend_cost_mode is only valid for problem='hh'.")
        if str(continuation_mode) != "phase3_v1":
            raise ValueError("phase3_backend_cost_mode is only valid for adapt_continuation_mode='phase3_v1'.")
        if int(phase3_backend_optimization_level) not in {0, 1, 2, 3}:
            raise ValueError("phase3_backend_optimization_level must be one of {0,1,2,3}.")
        if phase3_backend_cost_mode_key == "transpile_single_v1":
            if phase3_backend_name in {None, ""}:
                raise ValueError("transpile_single_v1 requires --phase3-backend-name.")
            if phase3_backend_shortlist_tokens:
                raise ValueError("transpile_single_v1 does not accept --phase3-backend-shortlist.")
        if phase3_backend_cost_mode_key == "transpile_shortlist_v1":
            if phase3_backend_name not in {None, ""}:
                raise ValueError("transpile_shortlist_v1 does not accept --phase3-backend-name.")
            if len(phase3_backend_shortlist_tokens) < 1:
                raise ValueError("transpile_shortlist_v1 requires --phase3-backend-shortlist.")
    pool_key_input = None if adapt_pool is None else str(adapt_pool).strip().lower()
    full_meta_class_filter_spec: HHFullMetaClassFilterSpec | None = None
    if adapt_pool_class_filter_json is not None:
        if str(problem_key) != "hh":
            raise ValueError("adapt_pool_class_filter_json is only valid for problem='hh'.")
        if pool_key_input != "full_meta":
            raise ValueError("adapt_pool_class_filter_json is only valid when adapt_pool='full_meta'.")
        full_meta_class_filter_spec = _load_hh_full_meta_class_filter_spec(Path(adapt_pool_class_filter_json))
        if (
            full_meta_class_filter_spec.source_num_sites is not None
            and int(full_meta_class_filter_spec.source_num_sites) != int(num_sites)
        ):
            raise ValueError(
                "HH full_meta class filter source_num_sites does not match this run: "
                f"got {full_meta_class_filter_spec.source_num_sites}, expected {num_sites}."
            )
        if (
            full_meta_class_filter_spec.source_n_ph_max is not None
            and int(full_meta_class_filter_spec.source_n_ph_max) != int(n_ph_max)
        ):
            raise ValueError(
                "HH full_meta class filter source_n_ph_max does not match this run: "
                f"got {full_meta_class_filter_spec.source_n_ph_max}, expected {n_ph_max}."
            )
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

    def _add_adapt_analytic_noise(value: float) -> float:
        value_f = float(value)
        if not adapt_analytic_noise_enabled:
            return value_f
        return value_f + float(
            adapt_noise_rng.normal(0.0, float(adapt_analytic_noise_std_val))
        )

    t0 = time.perf_counter()
    hf_bits = "N/A"
    _ai_log(
        "hardcoded_adapt_vqe_start",
        L=int(num_sites),
        problem=str(problem),
        adapt_pool=(str(pool_key_input) if pool_key_input is not None else None),
        adapt_pool_class_filter_json=(
            str(adapt_pool_class_filter_json) if adapt_pool_class_filter_json is not None else None
        ),
        adapt_pool_class_filter_classifier_version=(
            str(full_meta_class_filter_spec.classifier_version)
            if full_meta_class_filter_spec is not None
            else None
        ),
        adapt_pool_class_filter_keep_classes=(
            list(full_meta_class_filter_spec.keep_classes)
            if full_meta_class_filter_spec is not None
            else None
        ),
        adapt_continuation_mode=str(continuation_mode),
        phase3_motif_source_json=(str(phase3_motif_source_json) if phase3_motif_source_json is not None else None),
        phase3_symmetry_mitigation_mode=str(phase3_symmetry_mitigation_mode_key),
        phase3_runtime_split_mode=str(phase3_runtime_split_mode_key),
        phase3_backend_cost_mode=str(phase3_backend_cost_mode_key),
        phase3_backend_name=(None if phase3_backend_name in {None, ''} else str(phase3_backend_name)),
        phase3_backend_shortlist=[str(x) for x in phase3_backend_shortlist_tokens],
        phase3_backend_transpile_seed=int(phase3_backend_transpile_seed),
        phase3_backend_optimization_level=int(phase3_backend_optimization_level),
        phase3_enable_rescue=bool(phase3_enable_rescue),
        adapt_beam_live_branches_requested=int(beam_policy.live_branches_requested),
        adapt_beam_children_per_parent_requested=(
            int(beam_policy.children_per_parent_requested)
            if beam_policy.children_per_parent_requested is not None
            else None
        ),
        adapt_beam_terminated_keep_requested=(
            int(beam_policy.terminated_keep_requested)
            if beam_policy.terminated_keep_requested is not None
            else None
        ),
        adapt_beam_live_branches=int(beam_policy.live_branches_effective),
        adapt_beam_children_per_parent=int(beam_policy.children_per_parent_effective),
        adapt_beam_terminated_keep=int(beam_policy.terminated_keep_effective),
        adapt_beam_enabled=bool(beam_policy.beam_enabled),
        max_depth=int(max_depth),
        maxiter=int(maxiter),
        adapt_inner_optimizer=str(adapt_inner_optimizer_key),
        adapt_analytic_noise_std=float(adapt_analytic_noise_std_val),
        adapt_analytic_noise_seed=adapt_analytic_noise_seed_val,
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
        adapt_ref_json=(str(adapt_ref_json) if adapt_ref_json is not None else None),
    )

    num_particles = half_filled_num_particles(int(num_sites))
    psi_ref, _, _ = _default_adapt_input_state(
        problem=str(problem_key),
        num_sites=int(num_sites),
        ordering=str(ordering),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
    )
    if problem_key != "hh":
        hf_bits = str(
            hartree_fock_bitstring(
                n_sites=int(num_sites),
                num_particles=num_particles,
                indexing=str(ordering),
            )
        )
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
    full_meta_class_filter_meta: dict[str, Any] | None = None

    def _build_hh_pool_by_key(pool_key_hh: str) -> tuple[list[AnsatzTerm], str]:
        nonlocal full_meta_class_filter_meta
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
            if full_meta_class_filter_spec is not None:
                pool_full, full_meta_class_filter_meta = _filter_hh_full_meta_pool_by_class(
                    pool_full,
                    full_meta_class_filter_spec,
                )
                _ai_log(
                    "hardcoded_adapt_full_meta_class_filter_applied",
                    **dict(full_meta_class_filter_meta),
                )
            return list(pool_full), "hardcoded_adapt_vqe_full_meta"
        if key == "pareto_lean":
            pool_lean, lean_sizes = _build_hh_pareto_lean_pool(
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
                "hardcoded_adapt_pareto_lean_pool_built",
                **lean_sizes,
                dedup_total=int(len(pool_lean)),
            )
            return list(pool_lean), "hardcoded_adapt_vqe_pareto_lean"
        if key == "pareto_lean_l3":
            pool_lean_l3, lean_l3_sizes = _build_hh_pareto_lean_l3_pool(
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
                "hardcoded_adapt_pareto_lean_l3_pool_built",
                **lean_l3_sizes,
                dedup_total=int(len(pool_lean_l3)),
            )
            return list(pool_lean_l3), "hardcoded_adapt_vqe_pareto_lean_l3"
        if key == "pareto_lean_l2":
            pool_lean_l2, lean_l2_sizes = _build_hh_pareto_lean_l2_pool(
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
                "hardcoded_adapt_pareto_lean_l2_pool_built",
                **lean_l2_sizes,
                dedup_total=int(len(pool_lean_l2)),
            )
            return list(pool_lean_l2), "hardcoded_adapt_vqe_pareto_lean_l2"
        if key == "pareto_lean_gate_pruned":
            pool_gp, gp_sizes = _build_hh_pareto_lean_gate_pruned_pool(
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
                "hardcoded_adapt_pareto_lean_gate_pruned_pool_built",
                **gp_sizes,
                dedup_total=int(len(pool_gp)),
            )
            return list(pool_gp), "hardcoded_adapt_vqe_pareto_lean_gate_pruned"
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
        if key in {
            "paop",
            "paop_min",
            "paop_std",
            "paop_full",
            "paop_lf",
            "paop_lf_std",
            "paop_lf2_std",
            "paop_lf3_std",
            "paop_lf4_std",
            "paop_lf_full",
            "paop_sq_std",
            "paop_sq_full",
            "paop_bond_disp_std",
            "paop_hop_sq_std",
            "paop_pair_sq_std",
        }:
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
        if key in {"vlf_only", "sq_only", "vlf_sq", "sq_dens_only", "vlf_sq_dens"}:
            vlf_pool, _vlf_meta = _build_vlf_sq_pool(
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
            return list(vlf_pool), f"hardcoded_adapt_vqe_{key}"
        if key == "full_hamiltonian":
            return _build_full_hamiltonian_pool(h_poly, normalize_coeff=True), "hardcoded_adapt_vqe_full_hamiltonian_hh"
        raise ValueError(
            "For problem='hh', supported ADAPT pools are: "
            "hva, full_meta, pareto_lean, pareto_lean_l2, pareto_lean_gate_pruned, uccsd_paop_lf_full, paop, paop_min, paop_std, paop_full, "
            "paop_lf, paop_lf_std, paop_lf2_std, paop_lf_full, full_hamiltonian"
        )

    pool_stage_family: list[str] = []
    pool_family_ids: list[str] = []
    phase1_core_limit = 0
    phase1_residual_indices: set[int] = set()
    phase1_depth0_full_meta_override = False
    if continuation_mode in {"phase1_v1", "phase2_v1", "phase3_v1"} and problem_key == "hh":
        if pool_key_input in ("full_meta", "pareto_lean", "pareto_lean_l2", "pareto_lean_gate_pruned"):
            phase1_depth0_full_meta_override = True
            pool, _pool_method = _build_hh_pool_by_key(str(pool_key_input))
            phase1_core_limit = int(len(pool))
            phase1_residual_indices = set()
            pool_stage_family = ["core"] * int(len(pool))
            pool_family_ids = [str(pool_key_input)] * int(len(pool))
            _ai_log(
                "hardcoded_adapt_phase1_depth0_full_meta_override",
                continuation_mode=str(continuation_mode),
                pool_key=str(pool_key_input),
                pool_size=int(len(pool)),
            )
        else:
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

    qpb = int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding))) if problem_key == "hh" else 1
    pool_symmetry_specs: list[dict[str, Any] | None] = [None] * int(len(pool))
    pool_generator_registry: dict[str, dict[str, Any]] = {}
    if problem_key == "hh" and len(pool) > 0:
        base_pool_symmetry_specs = [
            dict(
                build_symmetry_spec(
                    family_id=str(pool_family_ids[idx] if idx < len(pool_family_ids) else "unknown"),
                    mitigation_mode=str(phase3_symmetry_mitigation_mode_key),
                ).__dict__
            )
            for idx in range(len(pool))
        ]
        raw_pool_generator_registry = build_pool_generator_registry(
            terms=pool,
            family_ids=pool_family_ids,
            num_sites=int(num_sites),
            ordering=str(ordering),
            qpb=int(max(1, qpb)),
            symmetry_specs=base_pool_symmetry_specs,
            split_policy=("deliberate_split" if bool(paop_split_paulis) else "preserve"),
        )
        filtered_pool: list[AnsatzTerm] = []
        filtered_stage_family: list[str] = []
        filtered_family_ids: list[str] = []
        filtered_specs: list[dict[str, Any] | None] = []
        filtered_registry: dict[str, dict[str, Any]] = {}
        removed_labels: list[str] = []
        removed_family_ids: list[str] = []
        for idx, term in enumerate(pool):
            label = str(term.label)
            meta = raw_pool_generator_registry.get(label)
            spec = (
                meta.get("symmetry_spec")
                if isinstance(meta, Mapping)
                else base_pool_symmetry_specs[idx]
            )
            if isinstance(spec, Mapping) and bool(spec.get("hard_guard", False)):
                removed_labels.append(label)
                removed_family_ids.append(str(pool_family_ids[idx] if idx < len(pool_family_ids) else "unknown"))
                continue
            filtered_pool.append(term)
            filtered_stage_family.append(str(pool_stage_family[idx] if idx < len(pool_stage_family) else pool_key))
            filtered_family_ids.append(str(pool_family_ids[idx] if idx < len(pool_family_ids) else pool_key))
            filtered_specs.append(dict(spec) if isinstance(spec, Mapping) else None)
            if isinstance(meta, Mapping):
                filtered_registry[label] = dict(meta)
        if removed_labels:
            _ai_log(
                "hardcoded_adapt_hh_pool_symmetry_filtered",
                removed_count=int(len(removed_labels)),
                kept_count=int(len(filtered_pool)),
                removed_labels_sample=[str(x) for x in removed_labels[:12]],
                removed_families_sample=[str(x) for x in removed_family_ids[:12]],
            )
        pool = filtered_pool
        pool_stage_family = filtered_stage_family
        pool_family_ids = filtered_family_ids
        pool_symmetry_specs = filtered_specs
        pool_generator_registry = filtered_registry
        if continuation_mode in {"phase1_v1", "phase2_v1", "phase3_v1"}:
            phase1_core_limit = int(sum(1 for stage in pool_stage_family if str(stage) == "core"))
            phase1_residual_indices = {
                int(idx) for idx, stage in enumerate(pool_stage_family) if str(stage) == "residual"
            }

    if len(pool) == 0:
        raise ValueError(f"ADAPT pool '{pool_key}' produced no operators for problem='{problem_key}'.")

    seq2p_logical_mode = bool(
        problem_key == "hh"
        and pool_key in _HH_UCCSD_PAOP_PRODUCT_SPECS
        and str(_HH_UCCSD_PAOP_PRODUCT_SPECS[str(pool_key)]["parameterization"]) == "double_sequential"
    )
    logical_candidates = (
        _build_seq2p_logical_candidates(pool, family_id=str(pool_key))
        if seq2p_logical_mode
        else []
    )
    _ai_log(
        "hardcoded_adapt_pool_built",
        pool_type=str(pool_key),
        pool_size=int(len(pool)),
        continuation_mode=str(continuation_mode),
        phase1_depth0_full_meta_override=bool(phase1_depth0_full_meta_override),
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
    phase3_split_events: list[dict[str, Any]] = []
    phase3_input_motif_library: dict[str, Any] | None = None
    phase3_runtime_split_summary: dict[str, Any] = {
        "mode": (str(phase3_runtime_split_mode_key) if phase3_enabled else "off"),
        "probed_parent_count": 0,
        "evaluated_child_count": 0,
        "rejected_child_count_symmetry": 0,
        "admissible_child_set_count": 0,
        "probe_parent_win_count": 0,
        "probe_child_set_count": 0,
        "selected_child_set_count": 0,
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
    if phase3_enabled and not pool_generator_registry:
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
    phase3_enable_rescue_requested = bool(phase3_enable_rescue)
    phase3_enable_rescue_effective = bool(
        phase3_enable_rescue_requested and (not phase3_oracle_inner_objective_enabled)
    )
    if phase3_enabled and phase3_motif_source_json is not None:
        phase3_input_motif_library = load_motif_library_from_json(Path(phase3_motif_source_json))
        if phase3_input_motif_library is not None:
            phase3_motif_usage["enabled"] = True
            phase3_motif_usage["source_tag"] = str(phase3_input_motif_library.get("source_tag", "payload"))
    if phase3_enabled and (
        bool(phase3_enable_rescue_requested) or bool(phase3_oracle_inner_objective_enabled)
    ):
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

    if str(problem).strip().lower() == "hh" and int(num_sites) <= 2:
        selected_parameterization_mode = "per_pauli_term"
    else:
        selected_parameterization_mode = "logical_shared"

    def _build_compiled_executor(ops: list[AnsatzTerm]) -> CompiledAnsatzExecutor:
        return CompiledAnsatzExecutor(
            ops,
            coefficient_tolerance=1e-12,
            ignore_identity=True,
            sort_terms=True,
            pauli_action_cache=pauli_action_cache,
            parameterization_mode=selected_parameterization_mode,
        )

    def _executor_theta_for_selected_state(
        theta_now: np.ndarray,
        layout_now: AnsatzParameterLayout,
        executor_now: CompiledAnsatzExecutor,
    ) -> np.ndarray:
        theta_arr = np.asarray(theta_now, dtype=float).reshape(-1)
        mode = str(
            getattr(executor_now, "parameterization_mode", selected_parameterization_mode)
        ).strip().lower()
        if mode == "logical_shared":
            return np.asarray(_logical_theta_alias(theta_arr, layout_now), dtype=float)
        return theta_arr

    def _prepare_selected_state(
        *,
        ops_now: Sequence[AnsatzTerm],
        theta_now: np.ndarray,
        executor_now: CompiledAnsatzExecutor | None,
        parameter_layout_now: AnsatzParameterLayout | None,
    ) -> np.ndarray:
        if len(ops_now) == 0:
            return np.array(psi_ref, copy=True)
        layout_now = (
            parameter_layout_now
            if parameter_layout_now is not None
            else _build_selected_layout(list(ops_now))
        )
        if adapt_state_backend_key == "compiled":
            executor_use = executor_now if executor_now is not None else _build_compiled_executor(list(ops_now))
            theta_exec = _executor_theta_for_selected_state(
                np.asarray(theta_now, dtype=float),
                layout_now,
                executor_use,
            )
            return executor_use.prepare_state(theta_exec, psi_ref)
        return _prepare_adapt_state(
            psi_ref,
            list(ops_now),
            np.asarray(theta_now, dtype=float),
            parameter_layout=layout_now,
        )

    def _build_selected_layout(ops: list[AnsatzTerm]) -> AnsatzParameterLayout:
        return build_parameter_layout(
            ops,
            ignore_identity=True,
            coefficient_tolerance=1e-12,
            sort_terms=True,
        )

    # ADAPT-VQE main loop
    selected_ops: list[AnsatzTerm] = []
    selected_layout = _build_selected_layout(selected_ops)
    theta = np.zeros(0, dtype=float)
    selected_executor: CompiledAnsatzExecutor | None = None
    history: list[dict[str, Any]] = []
    nfev_total = 0
    stop_reason = "max_depth"

    scipy_minimize = None
    if adapt_inner_optimizer_key in {"COBYLA", "POWELL"}:
        from scipy.optimize import minimize as scipy_minimize

    def _scipy_inner_options(maxiter_value: int) -> dict[str, Any]:
        if adapt_inner_optimizer_key == "COBYLA":
            return {"maxiter": int(maxiter_value), "rhobeg": 0.3}
        if adapt_inner_optimizer_key == "POWELL":
            return {"maxiter": int(maxiter_value), "xtol": 1e-4, "ftol": 1e-8}
        raise ValueError(f"Unsupported SciPy ADAPT inner optimizer: {adapt_inner_optimizer_key}")

    # Pool availability tracking (for no-repeat mode)
    available_indices = (
        set(range(int(phase1_core_limit)))
        if phase1_enabled
        else set(range(len(pool)))
    )
    selection_counts = np.zeros(len(pool), dtype=np.int64)
    logical_available_indices = (
        set(range(len(logical_candidates)))
        if seq2p_logical_mode
        else set()
    )
    logical_selection_counts = (
        np.zeros(len(logical_candidates), dtype=np.int64)
        if seq2p_logical_mode
        else np.zeros(0, dtype=np.int64)
    )
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
        cap_phase1_min=int(max(1, phase1_shortlist_size_val)),
        cap_phase1_max=int(max(1, phase1_shortlist_size_val)),
        cap_phase2_min=int(max(1, phase2_shortlist_size)),
        cap_phase2_max=int(max(1, phase2_shortlist_size)),
        cap_phase3_min=int(max(1, phase2_shortlist_size)),
        cap_phase3_max=int(max(1, phase2_shortlist_size)),
        shot_min=1,
        shot_max=1,
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
        wD=float(phase1_lambda_compile),
        wG=float(phase1_lambda_measure),
        wC=float(phase1_lambda_measure),
        wc=float(phase1_lambda_measure),
        depth_ref=float(phase1_depth_ref_val),
        group_ref=float(phase1_group_ref_val),
        shot_ref=float(phase1_shot_ref_val),
        family_ref=float(phase1_family_ref_val),
        compile_cx_proxy_weight=float(phase1_compile_cx_proxy_weight),
        compile_sq_proxy_weight=float(phase1_compile_sq_proxy_weight),
        compile_rotation_step_weight=float(phase1_compile_rotation_step_weight),
        compile_position_shift_weight=float(phase1_compile_position_shift_weight),
        compile_refit_active_weight=float(phase1_compile_refit_active_weight),
        measure_groups_weight=float(phase1_measure_groups_weight),
        measure_shots_weight=float(phase1_measure_shots_weight),
        measure_reuse_weight=float(phase1_measure_reuse_weight),
        opt_dim_cost_scale=float(phase1_opt_dim_cost_scale),
        family_repeat_cost_scale=float(phase1_family_repeat_cost_scale),
        lifetime_cost_mode=(
            str(phase3_lifetime_cost_mode_key)
            if phase3_enabled and str(phase3_lifetime_cost_mode_key) != "off"
            else "off"
        ),
    )
    phase1_compile_oracle = Phase1CompileCostOracle()
    phase1_measure_cache = MeasurementCacheAudit(nominal_shots_per_group=1)
    backend_compile_cfg = BackendCompileConfig(
        mode=str(phase3_backend_cost_mode_key),
        requested_backend_name=(None if phase3_backend_name in {None, ''} else str(phase3_backend_name)),
        requested_backend_shortlist=tuple(str(x) for x in phase3_backend_shortlist_tokens),
        seed_transpiler=int(phase3_backend_transpile_seed),
        optimization_level=int(phase3_backend_optimization_level),
    )
    backend_compile_oracle = (
        BackendCompileOracle(
            config=backend_compile_cfg,
            num_qubits=int(round(math.log2(psi_ref.size))),
            ref_state=np.asarray(psi_ref, dtype=complex),
        )
        if str(phase3_backend_cost_mode_key) != "proxy"
        else None
    )
    if backend_compile_oracle is not None and len(getattr(backend_compile_oracle, "targets", ())) == 0:
        raise RuntimeError("No backend targets could be resolved for phase3 backend-aware scoring.")
    phase2_score_cfg = FullScoreConfig(
        z_alpha=float(phase2_score_z_alpha_val),
        lambda_F=float(phase2_lambda_F_val),
        lambda_H=float(max(1e-12, phase2_lambda_H)),
        rho=float(max(1e-6, phase2_rho)),
        gamma_N=float(max(0.0, phase2_gamma_N)),
        depth_ref=float(phase2_depth_ref_val),
        group_ref=float(phase2_group_ref_val),
        shot_ref=float(phase2_shot_ref_val),
        optdim_ref=float(phase2_optdim_ref_val),
        reuse_ref=float(phase2_reuse_ref_val),
        family_ref=float(phase2_family_ref_val),
        novelty_eps=float(phase2_novelty_eps_val),
        cheap_score_eps=float(phase2_cheap_score_eps_val),
        leakage_cap=float(phase2_leakage_cap_val),
        metric_floor=float(phase2_metric_floor_val),
        reduced_metric_collapse_rel_tol=float(phase2_reduced_metric_collapse_rel_tol_val),
        ridge_growth_factor=float(phase2_ridge_growth_factor_val),
        ridge_max_steps=int(phase2_ridge_max_steps_val),
        wD=float(max(0.0, phase2_w_depth)),
        wG=float(max(0.0, phase2_w_group)),
        wC=float(max(0.0, phase2_w_shot)),
        wP=float(max(0.0, phase2_w_optdim)),
        wc=float(max(0.0, phase2_w_reuse)),
        lifetime_weight=float(max(0.0, phase2_w_lifetime)),
        eta_L=float(max(0.0, phase2_eta_L)),
        compile_cx_proxy_weight=float(phase2_compile_cx_proxy_weight),
        compile_sq_proxy_weight=float(phase2_compile_sq_proxy_weight),
        compile_rotation_step_weight=float(phase2_compile_rotation_step_weight),
        compile_position_shift_weight=float(phase2_compile_position_shift_weight),
        compile_refit_active_weight=float(phase2_compile_refit_active_weight),
        measure_groups_weight=float(phase2_measure_groups_weight),
        measure_shots_weight=float(phase2_measure_shots_weight),
        measure_reuse_weight=float(phase2_measure_reuse_weight),
        opt_dim_cost_scale=float(phase2_opt_dim_cost_scale),
        family_repeat_cost_scale=float(phase2_family_repeat_cost_scale),
        motif_bonus_weight=float(max(0.0, phase2_motif_bonus_weight)),
        duplicate_penalty_weight=float(max(0.0, phase2_duplicate_penalty_weight)),
        shortlist_fraction=float(max(0.05, phase2_shortlist_fraction)),
        shortlist_size=int(max(1, phase2_shortlist_size)),
        phase2_frontier_ratio=float(max(0.0, min(1.0, phase2_frontier_ratio))),
        phase3_frontier_ratio=float(max(0.0, min(1.0, phase3_frontier_ratio))),
        batch_target_size=int(max(1, phase2_batch_target_size)),
        batch_size_cap=int(max(1, phase2_batch_size_cap)),
        batch_near_degenerate_ratio=float(max(0.0, min(1.0, phase2_batch_near_degenerate_ratio))),
        batch_rank_rel_tol=float(max(0.0, phase2_batch_rank_rel_tol)),
        batch_additivity_tol=float(max(0.0, phase2_batch_additivity_tol)),
        compat_overlap_weight=float(max(0.0, phase2_compat_overlap_weight)),
        compat_comm_weight=float(max(0.0, phase2_compat_comm_weight)),
        compat_curv_weight=float(max(0.0, phase2_compat_curv_weight)),
        compat_sched_weight=float(max(0.0, phase2_compat_sched_weight)),
        compat_measure_weight=float(max(0.0, phase2_compat_measure_weight)),
        lifetime_cost_mode=(
            str(phase3_lifetime_cost_mode_key)
            if phase3_enabled and str(phase3_lifetime_cost_mode_key) != "off"
            else "off"
        ),
        remaining_evaluations_proxy_mode=(
            (
                "remaining_depth"
                if phase3_enabled and str(phase3_lifetime_cost_mode_key) != "off"
                else "none"
            )
            if str(phase2_remaining_evaluations_proxy_mode).strip().lower() == "auto"
            else str(phase2_remaining_evaluations_proxy_mode).strip().lower()
        ),
    )
    if phase3_enabled and float(phase2_score_cfg.lambda_F) <= 0.0:
        raise ValueError("phase3_v1 cheap ratio scoring requires phase2_lambda_F > 0.")
    phase2_novelty_oracle = Phase2NoveltyOracle()
    phase2_curvature_oracle = Phase2CurvatureOracle()
    phase2_memory_adapter = Phase2OptimizerMemoryAdapter()
    phase2_compiled_term_cache: dict[str, Any] = {}
    phase2_optimizer_memory = phase2_memory_adapter.unavailable(
        method=str(adapt_inner_optimizer_key),
        parameter_count=int(theta.size),
        reason="pre_seed_state",
    )

    def _controller_snapshot_dict(snapshot: Any | None) -> dict[str, Any] | None:
        if snapshot is None:
            return None
        return {
            "step_index": int(getattr(snapshot, "step_index", 0)),
            "depth_local": int(getattr(snapshot, "depth_local", 0)),
            "depth_left": int(getattr(snapshot, "depth_left", 0)),
            "runway_ratio": float(getattr(snapshot, "runway_ratio", 0.0)),
            "early_coordinate": float(getattr(snapshot, "early_coordinate", 0.0)),
            "late_coordinate": float(getattr(snapshot, "late_coordinate", 0.0)),
            "frontier_ratio": float(getattr(snapshot, "frontier_ratio", 1.0)),
            "phase_thresholds": dict(getattr(snapshot, "phase_thresholds", {})),
            "phase_caps": dict(getattr(snapshot, "phase_caps", {})),
            "phase_shots": dict(getattr(snapshot, "phase_shots", {})),
            "phase_uncertainty": dict(getattr(snapshot, "phase_uncertainty", {})),
            "snapshot_version": str(getattr(snapshot, "snapshot_version", "phase123_controller_v1")),
        }

    def _selector_score_value(row: Mapping[str, Any] | None) -> float:
        if not isinstance(row, Mapping):
            return 0.0
        for key in ("selector_score", "full_v2_score", "phase2_raw_score", "cheap_score", "simple_score"):
            raw = row.get(key)
            if raw is None:
                continue
            try:
                return float(raw)
            except (TypeError, ValueError):
                continue
        return 0.0

    def _selector_burden_value(row: Mapping[str, Any] | None) -> float:
        if not isinstance(row, Mapping):
            return 0.0
        for key in ("selector_burden", "phase3_burden_total", "phase2_burden_total", "cheap_burden_total"):
            raw = row.get(key)
            if raw is None:
                continue
            try:
                return float(raw)
            except (TypeError, ValueError):
                continue
        return 0.0

    def _attach_controller_snapshot(
        records: Sequence[Mapping[str, Any]],
        *,
        snapshot: Any | None,
    ) -> list[dict[str, Any]]:
        snapshot_dict = _controller_snapshot_dict(snapshot)
        out: list[dict[str, Any]] = []
        for rec in records:
            updated = dict(rec)
            feat_obj = updated.get("feature")
            if isinstance(feat_obj, CandidateFeatures) and snapshot_dict is not None:
                updated["feature"] = CandidateFeatures(
                    **{
                        **feat_obj.__dict__,
                        "controller_snapshot": dict(snapshot_dict),
                    }
                )
                updated.update(
                    {
                        "simple_score": float(updated["feature"].simple_score or updated.get("simple_score", float("-inf"))),
                        "cheap_score": float(updated["feature"].cheap_score or updated.get("cheap_score", float("-inf"))),
                        "phase2_raw_score": float(updated["feature"].phase2_raw_score or updated.get("phase2_raw_score", float("-inf"))),
                        "full_v2_score": float(updated["feature"].full_v2_score or updated.get("full_v2_score", float("-inf"))),
                    }
                )
            out.append(updated)
        return out

    def _phase1_shortlist_score_key() -> str:
        return "cheap_score" if bool(phase2_enabled) else "simple_score"

    def _phase_shortlist_with_legacy_hook(
        records: Sequence[Mapping[str, Any]],
        *,
        score_key: str,
        threshold: float,
        cap: int,
        frontier_ratio: float,
        tie_break_score_key: str | None = None,
        shortlist_flag: str | None = None,
    ) -> list[dict[str, Any]]:
        records_list = [dict(rec) for rec in records]
        if records_list:
            legacy_cfg = replace(
                phase2_score_cfg,
                shortlist_fraction=1.0,
                shortlist_size=max(1, len(records_list)),
            )
            shortlist_records(
                records_list,
                cfg=legacy_cfg,
                score_key=score_key,
                tie_break_score_key=tie_break_score_key,
            )
        shortlisted = phase_shortlist_records(
            records_list,
            score_key=score_key,
            threshold=threshold,
            cap=cap,
            frontier_ratio=frontier_ratio,
            tie_break_score_key=tie_break_score_key,
            shortlist_flag=shortlist_flag,
        )
        if shortlisted:
            return shortlisted
        if not records_list:
            return []
        return phase_shortlist_records(
            records_list,
            score_key=score_key,
            threshold=float("-inf"),
            cap=1,
            frontier_ratio=0.0,
            tie_break_score_key=tie_break_score_key,
            shortlist_flag=shortlist_flag,
        )

    def _selection_record_key(rec: Mapping[str, Any]) -> tuple[str, int, int]:
        return (
            str(rec.get("candidate_label") or getattr(rec.get("candidate_term"), "label", "")),
            int(rec.get("candidate_pool_index", -1)),
            int(rec.get("position_id", -1)),
        )

    def _positive_full_v2_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [
            dict(rec)
            for rec in records
            if float(rec.get("full_v2_score", float("-inf"))) > 0.0
        ]

    def _selection_pool_from_shortlist(
        shortlist_records_in: Sequence[Mapping[str, Any]],
        full_records_in: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        out = [dict(rec) for rec in shortlist_records_in]
        seen = {_selection_record_key(rec) for rec in out}
        positive_full_records = _positive_full_v2_records(full_records_in)
        fallback_candidates = positive_full_records if positive_full_records else [dict(rec) for rec in full_records_in[:1]]
        for rec in fallback_candidates[:1]:
            rec_key = _selection_record_key(rec)
            if rec_key not in seen:
                out.append(dict(rec))
                seen.add(rec_key)
        return sorted(out, key=_phase2_record_sort_key)

    def _record_controller_snapshot(record: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(record, Mapping):
            return None
        snapshot_raw = record.get("controller_snapshot")
        if isinstance(snapshot_raw, Mapping):
            return dict(snapshot_raw)
        feat_obj = record.get("feature")
        snapshot_feat = (
            getattr(feat_obj, "controller_snapshot", None)
            if isinstance(feat_obj, CandidateFeatures)
            else None
        )
        if isinstance(snapshot_feat, Mapping):
            return dict(snapshot_feat)
        return None

    def _phase3_tie_beam_selection_pool(
        records: Sequence[Mapping[str, Any]],
        *,
        default_cap: int,
        score_key: str,
        score_ratio: float,
        abs_tol: float,
        max_branches: int,
        max_late_coordinate: float,
        min_depth_left: int,
        depth_one_based: int,
        max_depth_local: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        ordered = sorted([dict(rec) for rec in records], key=_phase2_record_sort_key)
        cap_default = int(max(1, default_cap))
        derived_depth_left = int(max(0, int(max_depth_local) - int(depth_one_based)))
        derived_late_coordinate = float(int(depth_one_based) / max(1, int(max_depth_local)))
        if not ordered:
            return [], {
                "active": False,
                "band_count": 0,
                "selected_count": 0,
                "best_score": float("-inf"),
                "depth_left": int(derived_depth_left),
                "late_coordinate": float(derived_late_coordinate),
                "reason": "empty",
            }
        best_score = float(ordered[0].get(score_key, float("-inf")))
        snapshot = _record_controller_snapshot(ordered[0])
        depth_left = int(
            max(
                0,
                int(snapshot.get("depth_left", derived_depth_left))
                if isinstance(snapshot, Mapping)
                else int(derived_depth_left),
            )
        )
        late_coordinate = float(
            snapshot.get("late_coordinate", derived_late_coordinate)
            if isinstance(snapshot, Mapping)
            else float(derived_late_coordinate)
        )
        criteria_enabled = bool(
            int(max_branches) > int(cap_default)
            and (
                (float(score_ratio) < 1.0 and math.isfinite(best_score) and best_score > 0.0)
                or (float(abs_tol) > 0.0 and math.isfinite(best_score))
            )
        )
        maturity_open = bool(
            int(depth_left) >= int(min_depth_left)
            and float(late_coordinate) <= float(max_late_coordinate)
        )
        if not criteria_enabled:
            return ordered[:cap_default], {
                "active": False,
                "band_count": int(min(len(ordered), cap_default)),
                "selected_count": int(min(len(ordered), cap_default)),
                "best_score": float(best_score),
                "depth_left": int(depth_left),
                "late_coordinate": float(late_coordinate),
                "reason": "disabled",
            }
        if not maturity_open:
            return ordered[:cap_default], {
                "active": False,
                "band_count": int(min(len(ordered), cap_default)),
                "selected_count": int(min(len(ordered), cap_default)),
                "best_score": float(best_score),
                "depth_left": int(depth_left),
                "late_coordinate": float(late_coordinate),
                "reason": "maturity_closed",
            }
        band: list[dict[str, Any]] = [dict(ordered[0])]
        seen = {_selection_record_key(ordered[0])}
        for rec in ordered[1:]:
            score_val = float(rec.get(score_key, float("-inf")))
            if not math.isfinite(score_val):
                continue
            within_ratio = bool(
                float(score_ratio) < 1.0 and score_val >= float(score_ratio) * best_score
            )
            within_abs = bool(
                float(abs_tol) > 0.0 and (best_score - score_val) <= float(abs_tol)
            )
            if not (within_ratio or within_abs):
                continue
            rec_key = _selection_record_key(rec)
            if rec_key in seen:
                continue
            band.append(dict(rec))
            seen.add(rec_key)
        if len(band) <= cap_default:
            return band[:cap_default], {
                "active": False,
                "band_count": int(len(band)),
                "selected_count": int(min(len(band), cap_default)),
                "best_score": float(best_score),
                "depth_left": int(depth_left),
                "late_coordinate": float(late_coordinate),
                "reason": "band_not_wider_than_default",
            }
        selected_cap = int(min(len(band), max(int(cap_default), int(max_branches))))
        return band[:selected_cap], {
            "active": True,
            "band_count": int(len(band)),
            "selected_count": int(selected_cap),
            "best_score": float(best_score),
            "depth_left": int(depth_left),
            "late_coordinate": float(late_coordinate),
            "reason": "phase3_score_band",
        }

    def _controller_cap(snapshot: Any | None, phase_name: str, default_value: int) -> int:
        if snapshot is None:
            return int(max(1, default_value))
        caps = getattr(snapshot, "phase_caps", {})
        return int(max(1, caps.get(str(phase_name), default_value)))

    def _controller_threshold(snapshot: Any | None, phase_name: str) -> float:
        if snapshot is None:
            return 0.0
        thresholds = getattr(snapshot, "phase_thresholds", {})
        return float(thresholds.get(str(phase_name), 0.0))

    phase1_residual_opened = False
    phase1_last_probe_reason = "none"
    phase1_last_positions_considered: list[int] = []
    phase1_last_trough_detected = False
    phase1_last_trough_probe_triggered = False
    phase1_last_selected_score: float | None = None
    phase1_features_history: list[dict[str, Any]] = []
    phase1_stage_events: list[dict[str, Any]] = []
    phase1_scaffold_pre_prune: dict[str, Any] | None = None
    phase1_prune_cfg = PruneConfig(
        max_candidates=int(max(1, phase1_prune_max_candidates)),
        min_candidates=1,
        fraction_candidates=float(max(0.0, phase1_prune_fraction)),
        max_regression=float(max(0.0, phase1_prune_max_regression)),
        retained_gain_ratio=0.5,
        protect_steps=2,
        stale_age=2,
        stagnation_threshold=0.0,
        small_theta_abs=1e-3,
        small_theta_relative=0.5,
        cooldown_steps=2,
        local_window_size=4,
        old_fraction=0.25,
    )
    phase1_prune_live_mode = bool(phase1_enabled and phase1_prune_enabled)
    phase1_prune_checkpoint_period = 3
    phase1_prune_maturity_threshold = 0.5
    phase1_prune_snr_threshold = 1.0

    def _default_prune_summary(*, reason: str, energy: float) -> dict[str, Any]:
        return {
            "enabled": bool(phase1_enabled and phase1_prune_enabled),
            "executed": False,
            "rolled_back": False,
            "permission_open": False,
            "permission_reason": str(reason),
            "mature_open": False,
            "runway_ratio": 1.0,
            "u_sat": 0.0,
            "maturity_threshold": float(phase1_prune_maturity_threshold),
            "checkpoint_due": False,
            "checkpoint_period": int(max(1, phase1_prune_checkpoint_period)),
            "sigma_phase3": 0.0,
            "gain_floor": 0.0,
            "gain_ewma_recent": 0.0,
            "snr_adm": 0.0,
            "snr_threshold": float(phase1_prune_snr_threshold),
            "snr_low_enough": True,
            "protect_steps": int(phase1_prune_cfg.protect_steps),
            "stale_age": int(phase1_prune_cfg.stale_age),
            "stagnation_threshold": float(phase1_prune_cfg.stagnation_threshold),
            "cooldown_steps": int(phase1_prune_cfg.cooldown_steps),
            "theta_small": float(max(0.0, phase1_prune_cfg.small_theta_abs)),
            "gate_rows": [],
            "stale_age_pass_indices": [],
            "stagnation_pass_indices": [],
            "protected_indices": [],
            "cooldown_blocked_indices": [],
            "mature_eligible_indices": [],
            "small_angle_pool_indices": [],
            "accepted_count": 0,
            "candidate_count": 0,
            "probe_indices": [],
            "probe_labels": [],
            "frozen_scores": [],
            "selected_index": None,
            "selected_label": None,
            "decisions": [],
            "trial": None,
            "metadata": [],
            "energy_before": float(energy),
            "energy_after_prune": float(energy),
            "energy_after_post_refit": float(energy),
            "post_refit_executed": False,
        }

    def _fallback_prune_metadata(*, label: str, theta_value: float) -> ScaffoldCoordinateMetadata:
        theta_abs = abs(float(theta_value))
        return ScaffoldCoordinateMetadata(
            candidate_label=str(label),
            generator_id=None,
            admission_step=0,
            first_seen_step=0,
            selector_score=0.0,
            selector_burden=0.0,
            cooldown_remaining=0,
            cumulative_abs_motion=0.0,
            recent_abs_motion=0.0,
            stagnation_score=float(1.0 / (1.0 + theta_abs + 1e-12)),
        )

    def _reconstruct_prune_first_seen_steps(history_rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        out: dict[str, int] = {}
        for step_idx, row in enumerate(history_rows, start=1):
            labels_raw = row.get("selected_ops")
            labels = (
                [str(x) for x in labels_raw]
                if isinstance(labels_raw, Sequence) and not isinstance(labels_raw, (str, bytes))
                else [str(row.get("selected_op", ""))]
            )
            for label in labels:
                out.setdefault(str(label), int(step_idx))
        return out

    def _initialize_prune_metadata_state(
        *,
        labels_now: list[str],
        theta_logical_now: np.ndarray,
    ) -> tuple[list[ScaffoldCoordinateMetadata], dict[str, int]]:
        entries: list[ScaffoldCoordinateMetadata] = []
        first_seen_by_label = _reconstruct_prune_first_seen_steps(history)
        for step_idx, row in enumerate(history, start=1):
            if not isinstance(row, Mapping):
                continue
            labels_step_raw = row.get("selected_ops")
            labels_step = (
                [str(x) for x in labels_step_raw]
                if isinstance(labels_step_raw, Sequence) and not isinstance(labels_step_raw, (str, bytes))
                else [str(row.get("selected_op", ""))]
            )
            positions_step_raw = row.get("selected_positions")
            positions_step = (
                [int(x) for x in positions_step_raw]
                if isinstance(positions_step_raw, Sequence) and not isinstance(positions_step_raw, (str, bytes))
                else [int(row.get("selected_position", len(entries)))]
            )
            feature_rows_raw = row.get("selected_feature_rows")
            feature_rows = (
                list(feature_rows_raw)
                if isinstance(feature_rows_raw, Sequence) and not isinstance(feature_rows_raw, (str, bytes))
                else []
            )
            original_positions_seen: list[int] = []
            for item_idx, label_step in enumerate(labels_step):
                pos_orig = int(positions_step[item_idx]) if item_idx < len(positions_step) else int(len(entries))
                pos_eff = int(pos_orig + sum(1 for prev in original_positions_seen if int(prev) <= int(pos_orig)))
                feature_row = feature_rows[item_idx] if item_idx < len(feature_rows) else row
                feature_mapping = feature_row if isinstance(feature_row, Mapping) else row
                meta = ScaffoldCoordinateMetadata(
                    candidate_label=str(label_step),
                    generator_id=(
                        str(feature_mapping.get("generator_id"))
                        if feature_mapping.get("generator_id") is not None
                        else None
                    ),
                    admission_step=int(step_idx),
                    first_seen_step=int(first_seen_by_label.setdefault(str(label_step), int(step_idx))),
                    selector_score=float(_selector_score_value(feature_mapping)),
                    selector_burden=float(_selector_burden_value(feature_mapping)),
                    cooldown_remaining=0,
                    cumulative_abs_motion=0.0,
                    recent_abs_motion=0.0,
                    stagnation_score=0.0,
                )
                entries.insert(max(0, min(int(pos_eff), len(entries))), meta)
                original_positions_seen.append(int(pos_orig))
        aligned: list[ScaffoldCoordinateMetadata] = []
        theta_vals = np.asarray(theta_logical_now, dtype=float).reshape(-1)
        theta_scale = float(max(np.median(np.abs(theta_vals)) if theta_vals.size > 0 else 0.0, 1e-12))
        cursor = 0
        for idx_now, label_now in enumerate(labels_now):
            match_idx = None
            for entry_idx in range(cursor, len(entries)):
                if str(entries[entry_idx].candidate_label) == str(label_now):
                    match_idx = int(entry_idx)
                    break
            base_meta = (
                entries[int(match_idx)]
                if match_idx is not None
                else _fallback_prune_metadata(
                    label=str(label_now),
                    theta_value=(float(theta_vals[idx_now]) if idx_now < int(theta_vals.size) else 0.0),
                )
            )
            if match_idx is not None:
                cursor = int(match_idx + 1)
            theta_abs = abs(float(theta_vals[idx_now])) if idx_now < int(theta_vals.size) else 0.0
            aligned.append(
                ScaffoldCoordinateMetadata(
                    **{
                        **base_meta.__dict__,
                        "first_seen_step": int(first_seen_by_label.setdefault(str(label_now), int(base_meta.first_seen_step))),
                        "stagnation_score": float(max(0.0, 1.0 - theta_abs / theta_scale)),
                    }
                )
            )
        return aligned, first_seen_by_label

    def _refresh_prune_metadata_stagnation(
        metadata_rows: Sequence[ScaffoldCoordinateMetadata],
        theta_logical_now: np.ndarray,
    ) -> list[ScaffoldCoordinateMetadata]:
        theta_vals = np.asarray(theta_logical_now, dtype=float).reshape(-1)
        theta_scale = float(max(np.median(np.abs(theta_vals)) if theta_vals.size > 0 else 0.0, 1e-12))
        out: list[ScaffoldCoordinateMetadata] = []
        for idx, meta in enumerate(metadata_rows):
            theta_abs = abs(float(theta_vals[idx])) if idx < int(theta_vals.size) else 0.0
            out.append(
                ScaffoldCoordinateMetadata(
                    **{
                        **meta.__dict__,
                        "stagnation_score": float(max(0.0, 1.0 - theta_abs / theta_scale)),
                    }
                )
            )
        return out

    def _transport_prune_metadata_after_admission(
        *,
        metadata_rows: Sequence[ScaffoldCoordinateMetadata],
        labels_added: Sequence[str],
        positions_added: Sequence[int],
        feature_rows_added: Sequence[Mapping[str, Any]],
        selector_step: int,
        first_seen_steps: Mapping[str, int],
    ) -> tuple[list[ScaffoldCoordinateMetadata], dict[str, int]]:
        out = [
            ScaffoldCoordinateMetadata(
                **{**meta.__dict__, "cooldown_remaining": int(max(0, meta.cooldown_remaining - 1))}
            )
            for meta in metadata_rows
        ]
        first_seen = {str(k): int(v) for k, v in first_seen_steps.items()}
        original_positions_seen: list[int] = []
        for item_idx, label in enumerate(labels_added):
            pos_orig = int(positions_added[item_idx]) if item_idx < len(positions_added) else int(len(out))
            pos_eff = int(pos_orig + sum(1 for prev in original_positions_seen if int(prev) <= int(pos_orig)))
            feature_mapping = (
                feature_rows_added[item_idx]
                if item_idx < len(feature_rows_added) and isinstance(feature_rows_added[item_idx], Mapping)
                else {}
            )
            first_seen_step = int(first_seen.setdefault(str(label), int(selector_step)))
            out.insert(
                max(0, min(int(pos_eff), len(out))),
                ScaffoldCoordinateMetadata(
                    candidate_label=str(label),
                    generator_id=(
                        str(feature_mapping.get("generator_id"))
                        if feature_mapping.get("generator_id") is not None
                        else None
                    ),
                    admission_step=int(selector_step),
                    first_seen_step=int(first_seen_step),
                    selector_score=float(_selector_score_value(feature_mapping)),
                    selector_burden=float(_selector_burden_value(feature_mapping)),
                    cooldown_remaining=0,
                    cumulative_abs_motion=0.0,
                    recent_abs_motion=0.0,
                    stagnation_score=0.0,
                ),
            )
            original_positions_seen.append(int(pos_orig))
        return out, first_seen

    def _prune_refit_window_indices_live(
        *,
        removal_index: int,
        metadata_rows: Sequence[ScaffoldCoordinateMetadata],
        n_plus: int,
    ) -> list[int]:
        n_after = int(max(0, n_plus - 1))
        if n_after <= 0:
            return []
        omega_eff = int(max(1, min(int(phase1_prune_cfg.local_window_size), n_after)))
        local_start = int(max(0, min(int(removal_index) - (omega_eff - 1) // 2, n_after - omega_eff)))
        local_window = list(range(int(local_start), int(local_start + omega_eff)))
        nonlocal_indices = [idx for idx in range(n_after) if idx not in set(local_window)]
        oldest_count = int(math.ceil(float(phase1_prune_cfg.old_fraction) * float(len(nonlocal_indices))))
        oldest_count = int(max(0, min(oldest_count, len(nonlocal_indices))))
        oldest_tail = (
            sorted(
                nonlocal_indices,
                key=lambda idx: (
                    int(metadata_rows[idx].admission_step),
                    int(metadata_rows[idx].first_seen_step),
                    str(metadata_rows[idx].candidate_label),
                ),
            )[:oldest_count]
            if oldest_count > 0
            else []
        )
        return sorted({int(x) for x in [*local_window, *oldest_tail]})

    phase1_prune_metadata_state: list[ScaffoldCoordinateMetadata] = []
    phase1_prune_first_seen_steps: dict[str, int] = {}
    prune_summary = _default_prune_summary(reason="pre_energy_init", energy=0.0)

    def _execute_live_mature_prune_pass(
        *,
        ops_now: list[AnsatzTerm],
        theta_now: np.ndarray,
        energy_now: float,
        optimizer_memory_now: dict[str, Any],
        metadata_rows: Sequence[ScaffoldCoordinateMetadata],
        first_seen_steps: Mapping[str, int],
        controller_snapshot: Mapping[str, Any] | None,
        selector_step: int,
        admitted_gain: float,
        history_rows: Sequence[Mapping[str, Any]] | None = None,
    ) -> tuple[list[AnsatzTerm], np.ndarray, float, dict[str, Any], list[ScaffoldCoordinateMetadata], dict[str, int], dict[str, Any]]:
        summary = _default_prune_summary(reason="live_after_admission", energy=float(energy_now))
        if (not phase1_prune_live_mode) or int(len(ops_now)) <= 1:
            return list(ops_now), np.asarray(theta_now, dtype=float), float(energy_now), dict(optimizer_memory_now), [ScaffoldCoordinateMetadata(**dict(x.__dict__)) for x in metadata_rows], {str(k): int(v) for k, v in first_seen_steps.items()}, summary

        snapshot = dict(controller_snapshot) if isinstance(controller_snapshot, Mapping) else {}
        runway_ratio = float(max(0.0, min(1.0, snapshot.get("runway_ratio", 1.0))))
        u_sat = float(max(0.0, min(1.0, 1.0 - runway_ratio)))
        phase_uncertainty = snapshot.get("phase_uncertainty", {})
        sigma_phase3 = float(phase_uncertainty.get("phase3", 0.0)) if isinstance(phase_uncertainty, Mapping) else 0.0
        history_source = list(history_rows) if history_rows is not None else list(history)
        recent_gains = [
            max(0.0, float(row.get("energy_before_opt", 0.0)) - float(row.get("energy_after_opt", 0.0)))
            for row in history_source[-4:]
            if isinstance(row, Mapping)
        ]
        gain_ewma_recent = 0.0
        for gain_value in recent_gains:
            gain_ewma_recent = (
                float(gain_value)
                if not math.isfinite(float(gain_ewma_recent)) or float(gain_ewma_recent) <= 0.0
                else float(0.5 * float(gain_ewma_recent) + 0.5 * float(gain_value))
            )
        gain_floor = float(max(float(phase1_stage_cfg.weak_drop_threshold), float(gain_ewma_recent), 1e-12))
        noise_scale = float((1.0 + max(0.0, sigma_phase3)) * gain_floor)
        snr_adm = float(max(0.0, admitted_gain) / max(noise_scale, 1e-12))
        checkpoint_due = bool(int(selector_step) % int(max(1, phase1_prune_checkpoint_period)) == 0)
        mature_open = bool(float(u_sat) >= float(phase1_prune_maturity_threshold))
        permission_open = bool(mature_open and (float(snr_adm) <= float(phase1_prune_snr_threshold) or checkpoint_due))
        summary["runway_ratio"] = float(runway_ratio)
        summary["u_sat"] = float(u_sat)
        summary["mature_open"] = bool(mature_open)
        summary["checkpoint_due"] = bool(checkpoint_due)
        summary["sigma_phase3"] = float(sigma_phase3)
        summary["gain_floor"] = float(gain_floor)
        summary["gain_ewma_recent"] = float(gain_ewma_recent)
        summary["snr_adm"] = float(snr_adm)
        summary["snr_threshold"] = float(phase1_prune_snr_threshold)
        summary["snr_low_enough"] = bool(float(snr_adm) <= float(phase1_prune_snr_threshold))
        summary["permission_open"] = bool(permission_open)
        summary["permission_reason"] = (
            "checkpoint" if checkpoint_due and mature_open else ("low_snr" if permission_open else ("immature" if not mature_open else "admitted_gain_resolved"))
        )

        labels_now = [str(op.label) for op in ops_now]
        layout_now = _build_selected_layout(ops_now)
        theta_runtime_now = np.asarray(theta_now, dtype=float)
        theta_logical_now = np.asarray(_logical_theta_alias(theta_runtime_now, layout_now), dtype=float)
        metadata_live = _refresh_prune_metadata_stagnation(
            [ScaffoldCoordinateMetadata(**dict(x.__dict__)) for x in metadata_rows],
            theta_logical_now,
        )
        summary["metadata"] = [dict(x.__dict__) for x in metadata_live]
        theta_abs_wrapped = [
            float(abs((float(theta_val) + math.pi) % (2.0 * math.pi) - math.pi))
            for theta_val in np.asarray(theta_logical_now, dtype=float).reshape(-1).tolist()
        ]
        stale_age_pass_indices: list[int] = []
        stagnation_pass_indices: list[int] = []
        protected_indices: list[int] = []
        cooldown_blocked_indices: list[int] = []
        mature_eligible_indices: list[int] = []
        gate_rows: list[dict[str, Any]] = []
        for idx_meta, meta in enumerate(metadata_live):
            first_seen_age = int(max(0, int(selector_step) - int(meta.first_seen_step)))
            admission_age = int(max(0, int(selector_step) - int(meta.admission_step)))
            stale_age_pass = bool(first_seen_age >= int(phase1_prune_cfg.stale_age))
            stagnation_pass = bool(float(meta.stagnation_score) >= float(phase1_prune_cfg.stagnation_threshold))
            protected = bool(admission_age < int(phase1_prune_cfg.protect_steps))
            cooldown_blocked = bool(int(meta.cooldown_remaining) > 0)
            mature_eligible = bool(stale_age_pass and stagnation_pass and (not protected) and (not cooldown_blocked))
            if stale_age_pass:
                stale_age_pass_indices.append(int(idx_meta))
            if stagnation_pass:
                stagnation_pass_indices.append(int(idx_meta))
            if protected:
                protected_indices.append(int(idx_meta))
            if cooldown_blocked:
                cooldown_blocked_indices.append(int(idx_meta))
            if mature_eligible:
                mature_eligible_indices.append(int(idx_meta))
            gate_rows.append(
                {
                    "index": int(idx_meta),
                    "label": str(meta.candidate_label),
                    "theta_abs_wrapped": (
                        float(theta_abs_wrapped[idx_meta])
                        if idx_meta < len(theta_abs_wrapped)
                        else 0.0
                    ),
                    "first_seen_age": int(first_seen_age),
                    "admission_age": int(admission_age),
                    "cooldown_remaining": int(meta.cooldown_remaining),
                    "stagnation_score": float(meta.stagnation_score),
                    "selector_burden": float(meta.selector_burden),
                    "stale_age_pass": bool(stale_age_pass),
                    "stagnation_pass": bool(stagnation_pass),
                    "protected": bool(protected),
                    "cooldown_blocked": bool(cooldown_blocked),
                    "mature_eligible": bool(mature_eligible),
                }
            )
        theta_small = (
            float(
                max(
                    float(max(0.0, phase1_prune_cfg.small_theta_abs)),
                    float(max(0.0, phase1_prune_cfg.small_theta_relative))
                    * float(np.median([theta_abs_wrapped[i] for i in mature_eligible_indices])),
                )
            )
            if mature_eligible_indices
            else float(max(0.0, phase1_prune_cfg.small_theta_abs))
        )
        small_angle_pool_indices = [
            int(i)
            for i in mature_eligible_indices
            if float(theta_abs_wrapped[i]) <= float(theta_small) + 1e-15
        ]
        summary["protect_steps"] = int(phase1_prune_cfg.protect_steps)
        summary["stale_age"] = int(phase1_prune_cfg.stale_age)
        summary["stagnation_threshold"] = float(phase1_prune_cfg.stagnation_threshold)
        summary["cooldown_steps"] = int(phase1_prune_cfg.cooldown_steps)
        summary["theta_small"] = float(theta_small)
        summary["gate_rows"] = [dict(x) for x in gate_rows]
        summary["stale_age_pass_indices"] = [int(x) for x in stale_age_pass_indices]
        summary["stagnation_pass_indices"] = [int(x) for x in stagnation_pass_indices]
        summary["protected_indices"] = [int(x) for x in protected_indices]
        summary["cooldown_blocked_indices"] = [int(x) for x in cooldown_blocked_indices]
        summary["mature_eligible_indices"] = [int(x) for x in mature_eligible_indices]
        summary["small_angle_pool_indices"] = [int(x) for x in small_angle_pool_indices]
        if not permission_open:
            return list(ops_now), theta_runtime_now, float(energy_now), dict(optimizer_memory_now), metadata_live, {str(k): int(v) for k, v in first_seen_steps.items()}, summary

        prune_proxy_benefit = [
            float(meta.selector_score) / float(1.0 + max(0.0, meta.selector_burden))
            if math.isfinite(float(meta.selector_score))
            else float("inf")
            for meta in metadata_live
        ]
        candidate_indices = rank_prune_candidates(
            theta=np.asarray(theta_logical_now, dtype=float),
            labels=list(labels_now),
            marginal_proxy_benefit=list(prune_proxy_benefit),
            max_candidates=int(phase1_prune_cfg.max_candidates),
            min_candidates=int(phase1_prune_cfg.min_candidates),
            fraction_candidates=float(phase1_prune_cfg.fraction_candidates),
            selector_burden=[float(meta.selector_burden) for meta in metadata_live],
            admission_steps=[int(meta.admission_step) for meta in metadata_live],
            first_seen_steps=[int(meta.first_seen_step) for meta in metadata_live],
            cooldown_remaining=[int(meta.cooldown_remaining) for meta in metadata_live],
            stagnation_scores=[float(meta.stagnation_score) for meta in metadata_live],
            current_step=int(selector_step),
            protect_steps=int(phase1_prune_cfg.protect_steps),
            stale_age=int(phase1_prune_cfg.stale_age),
            stagnation_threshold=float(phase1_prune_cfg.stagnation_threshold),
            small_theta_abs=float(phase1_prune_cfg.small_theta_abs),
            small_theta_relative=float(phase1_prune_cfg.small_theta_relative),
        )
        summary["candidate_count"] = int(len(candidate_indices))
        summary["probe_indices"] = [int(x) for x in candidate_indices]
        summary["probe_labels"] = [str(labels_now[int(i)]) for i in candidate_indices]
        if not candidate_indices:
            return list(ops_now), theta_runtime_now, float(energy_now), dict(optimizer_memory_now), metadata_live, {str(k): int(v) for k, v in first_seen_steps.items()}, summary

        def _ops_from_labels(labels_cur: Sequence[str]) -> list[AnsatzTerm]:
            buckets: dict[str, list[AnsatzTerm]] = {}
            for op_ref in ops_now:
                buckets.setdefault(str(op_ref.label), []).append(op_ref)
            rebuilt: list[AnsatzTerm] = []
            for lbl in labels_cur:
                bucket = buckets.get(str(lbl), [])
                if bucket:
                    rebuilt.append(bucket.pop(0))
            return rebuilt

        def _refit_given_ops_live(
            ops_refit: list[AnsatzTerm],
            theta0: np.ndarray,
            active_logical_indices: list[int] | None,
            optimizer_memory_in: dict[str, Any],
        ) -> tuple[np.ndarray, float, dict[str, Any]]:
            if len(ops_refit) == 0:
                return np.zeros(0, dtype=float), float(energy_now), dict(optimizer_memory_in)
            layout_refit = _build_selected_layout(ops_refit)
            executor_refit = _build_compiled_executor(ops_refit) if adapt_state_backend_key == "compiled" else None
            theta_full = np.asarray(theta0, dtype=float).reshape(-1)
            active_runtime_indices = (
                list(range(int(layout_refit.runtime_parameter_count)))
                if active_logical_indices is None
                else runtime_indices_for_logical_indices(layout_refit, [int(i) for i in active_logical_indices])
            )
            def _obj_prune_live(x: np.ndarray) -> float:
                return _evaluate_selected_energy_objective(
                    ops_now=ops_refit,
                    theta_now=np.asarray(x, dtype=float),
                    executor_now=executor_refit,
                    parameter_layout_now=layout_refit,
                    objective_stage="prune_refit_live",
                    depth_marker=int(selector_step),
                )
            if not active_runtime_indices:
                return np.asarray(theta_full, dtype=float), float(_obj_prune_live(theta_full)), dict(optimizer_memory_in)
            _obj_reduced, opt_x0 = _make_reduced_objective(theta_full, active_runtime_indices, _obj_prune_live)
            if adapt_inner_optimizer_key == "SPSA":
                refit_memory = phase2_memory_adapter.select_active(
                    optimizer_memory_in,
                    active_indices=list(active_runtime_indices),
                    source="adapt.live_prune_refit.active_subset",
                ) if phase2_enabled else None
                res = spsa_minimize(
                    fun=_obj_reduced,
                    x0=opt_x0,
                    maxiter=int(max(25, min(int(maxiter), 120))),
                    seed=int(seed) + 800000 + int(len(ops_refit)),
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
                theta_out = np.asarray(theta_full, dtype=float).copy()
                result_x = np.asarray(res.x, dtype=float).ravel()
                for k, idx_active in enumerate(active_runtime_indices):
                    theta_out[int(idx_active)] = float(result_x[k])
                optimizer_out = dict(optimizer_memory_in)
                if phase2_enabled:
                    optimizer_out = phase2_memory_adapter.merge_active(
                        optimizer_memory_in,
                        active_indices=list(active_runtime_indices),
                        active_state=phase2_memory_adapter.from_result(
                            res,
                            method=str(adapt_inner_optimizer_key),
                            parameter_count=int(len(active_runtime_indices)),
                            source="adapt.live_prune_refit.result",
                        ),
                        source="adapt.live_prune_refit.merge",
                    )
                return theta_out, float(res.fun), optimizer_out
            res = _run_scipy_adapt_optimizer(
                method_key=str(adapt_inner_optimizer_key),
                objective=_obj_reduced,
                x0=opt_x0,
                maxiter=int(max(25, min(int(maxiter), 120))),
                context_label="live prune refit",
                scipy_minimize_fn=scipy_minimize,
            )
            theta_out = np.asarray(theta_full, dtype=float).copy()
            result_x = np.asarray(res.x, dtype=float).ravel()
            for k, idx_active in enumerate(active_runtime_indices):
                theta_out[int(idx_active)] = float(result_x[k])
            return theta_out, float(res.fun), dict(optimizer_memory_now)

        def _frozen_ablation_energy_live(idx_remove: int) -> tuple[float, np.ndarray]:
            runtime_remove_indices = runtime_indices_for_logical_indices(layout_now, [int(idx_remove)])
            ops_trial = list(ops_now)
            del ops_trial[int(idx_remove)]
            theta_trial0 = np.delete(theta_runtime_now, runtime_remove_indices)
            executor_trial = _build_compiled_executor(ops_trial) if adapt_state_backend_key == "compiled" and ops_trial else None
            energy_trial = _evaluate_selected_energy_objective(
                ops_now=ops_trial,
                theta_now=np.asarray(theta_trial0, dtype=float),
                executor_now=executor_trial,
                parameter_layout_now=_build_selected_layout(ops_trial),
                objective_stage="prune_frozen_live",
                depth_marker=int(selector_step),
            )
            return float(energy_trial), np.asarray(theta_trial0, dtype=float)

        best_candidate_index = None
        best_candidate_label = None
        best_trial_window_logical: list[int] = []
        best_frozen_score = float("inf")
        best_frozen_regression = float("inf")
        frozen_rows: list[dict[str, Any]] = []
        for idx_probe in candidate_indices:
            frozen_energy, _theta_frozen = _frozen_ablation_energy_live(int(idx_probe))
            frozen_regression = float(frozen_energy - float(energy_now))
            selector_burden = float(metadata_live[int(idx_probe)].selector_burden) if int(idx_probe) < len(metadata_live) else 0.0
            cheap_score_prune = cheap_prune_score(
                frozen_regression=float(frozen_regression),
                selector_burden=float(selector_burden),
            )
            metadata_after = [meta for meta_idx, meta in enumerate(metadata_live) if int(meta_idx) != int(idx_probe)]
            prune_window_logical = _prune_refit_window_indices_live(
                removal_index=int(idx_probe),
                metadata_rows=metadata_after,
                n_plus=int(len(ops_now)),
            )
            frozen_rows.append({
                "index": int(idx_probe),
                "label": str(labels_now[int(idx_probe)]),
                "frozen_energy": float(frozen_energy),
                "frozen_regression": float(frozen_regression),
                "selector_burden": float(selector_burden),
                "cheap_prune_score": float(cheap_score_prune),
                "refit_window_indices": [int(x) for x in prune_window_logical],
            })
            candidate_key = (float(cheap_score_prune), float(frozen_regression), int(idx_probe), str(labels_now[int(idx_probe)]))
            incumbent_key = (float(best_frozen_score), float(best_frozen_regression), int(best_candidate_index if best_candidate_index is not None else 10**9), str(best_candidate_label or ""))
            if best_candidate_index is None or candidate_key < incumbent_key:
                best_candidate_index = int(idx_probe)
                best_candidate_label = str(labels_now[int(idx_probe)])
                best_trial_window_logical = [int(x) for x in prune_window_logical]
                best_frozen_score = float(cheap_score_prune)
                best_frozen_regression = float(frozen_regression)
        summary["frozen_scores"] = [dict(x) for x in frozen_rows]
        summary["selected_index"] = int(best_candidate_index) if best_candidate_index is not None else None
        summary["selected_label"] = str(best_candidate_label) if best_candidate_label is not None else None
        retained_reference_energy = float(energy_now + max(0.0, admitted_gain))

        def _eval_with_removal_live(idx_remove: int, theta_cur: np.ndarray, labels_cur: list[str]) -> tuple[float, np.ndarray]:
            ops_current = _ops_from_labels(labels_cur)
            layout_current = _build_selected_layout(ops_current)
            runtime_remove_indices = runtime_indices_for_logical_indices(layout_current, [int(idx_remove)])
            ops_trial = list(ops_current)
            del ops_trial[int(idx_remove)]
            theta_trial0 = np.delete(np.asarray(theta_cur, dtype=float), runtime_remove_indices)
            theta_trial_opt, energy_trial, _optimizer_unused = _refit_given_ops_live(
                ops_trial,
                theta_trial0,
                [int(x) for x in best_trial_window_logical],
                dict(optimizer_memory_now),
            )
            return float(energy_trial), np.asarray(theta_trial_opt, dtype=float)

        theta_pruned, labels_pruned, prune_decisions, energy_after_prune = apply_pruning(
            theta=np.asarray(theta_runtime_now, dtype=float),
            labels=list(labels_now),
            candidate_indices=([int(best_candidate_index)] if best_candidate_index is not None else []),
            eval_with_removal=_eval_with_removal_live,
            energy_before=float(energy_now),
            max_regression=float(phase1_prune_cfg.max_regression),
            retained_reference_energy=float(retained_reference_energy),
            admitted_gain=float(max(0.0, admitted_gain)),
            retained_gain_ratio=float(phase1_prune_cfg.retained_gain_ratio),
        )
        accepted_count = int(sum(1 for d in prune_decisions if bool(d.accepted)))
        summary["executed"] = bool(best_candidate_index is not None)
        summary["accepted_count"] = int(accepted_count)
        summary["energy_after_prune"] = float(energy_after_prune)
        summary["energy_after_post_refit"] = float(energy_after_prune)
        summary["decisions"] = [dict(d.__dict__) for d in prune_decisions]
        summary["trial"] = dict(MaturePruneTrial(
            selector_step=int(selector_step),
            gate_open=bool(summary.get("permission_open", False)),
            probe_indices=[int(x) for x in candidate_indices],
            selected_index=(int(best_candidate_index) if best_candidate_index is not None else None),
            selected_label=(str(best_candidate_label) if best_candidate_label is not None else None),
            frozen_regression=(float(best_frozen_regression) if best_candidate_index is not None and math.isfinite(best_frozen_regression) else None),
            refit_energy=(float(energy_after_prune) if best_candidate_index is not None else None),
            retained_gain=(float(retained_reference_energy - float(energy_after_prune)) if best_candidate_index is not None else None),
            accepted=bool(accepted_count > 0),
            rollback_reason=(str(prune_decisions[0].reason) if prune_decisions and not bool(prune_decisions[0].accepted) else None),
        ).__dict__)

        metadata_out = [ScaffoldCoordinateMetadata(**dict(x.__dict__)) for x in metadata_live]
        first_seen_out = {str(k): int(v) for k, v in first_seen_steps.items()}
        optimizer_out = dict(optimizer_memory_now)
        if accepted_count > 0:
            accepted_remove_indices = [int(d.index) for d in prune_decisions if bool(d.accepted)]
            accepted_runtime_remove_indices = runtime_indices_for_logical_indices(layout_now, accepted_remove_indices)
            if phase2_enabled:
                optimizer_out = phase2_memory_adapter.remap_remove(
                    optimizer_out,
                    indices=list(accepted_runtime_remove_indices),
                )
            label_to_ops: dict[str, list[AnsatzTerm]] = {}
            for op in ops_now:
                label_to_ops.setdefault(str(op.label), []).append(op)
            rebuilt_ops: list[AnsatzTerm] = []
            for lbl in labels_pruned:
                bucket = label_to_ops.get(str(lbl), [])
                if bucket:
                    rebuilt_ops.append(bucket.pop(0))
            metadata_out = [
                meta for meta_idx, meta in enumerate(metadata_out)
                if int(meta_idx) not in set(accepted_remove_indices)
            ]
            return rebuilt_ops, np.asarray(theta_pruned, dtype=float), float(energy_after_prune), optimizer_out, metadata_out, first_seen_out, summary

        if best_candidate_index is not None and int(best_candidate_index) < len(metadata_out):
            cooled = dict(metadata_out[int(best_candidate_index)].__dict__)
            cooled["cooldown_remaining"] = int(phase1_prune_cfg.cooldown_steps)
            metadata_out[int(best_candidate_index)] = ScaffoldCoordinateMetadata(**cooled)
            summary["metadata"] = [dict(x.__dict__) for x in metadata_out]
        return list(ops_now), np.asarray(theta_runtime_now, dtype=float), float(energy_now), optimizer_out, metadata_out, first_seen_out, summary

    def _candidate_feature_rows(records: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not isinstance(records, Sequence):
            return out
        for rec in records:
            if not isinstance(rec, Mapping):
                continue
            feat_obj = rec.get("feature")
            if isinstance(feat_obj, CandidateFeatures):
                out.append(dict(feat_obj.__dict__))
            elif rec.get("candidate_label") is not None:
                out.append(dict(rec))
        return out

    phase1_last_retained_records: list[dict[str, Any]] = []
    phase2_last_shortlist_records: list[dict[str, Any]] = []
    phase2_last_geometric_shortlist_records: list[dict[str, Any]] = []
    phase2_last_retained_shortlist_records: list[dict[str, Any]] = []
    phase2_last_admitted_records: list[dict[str, Any]] = []
    phase2_last_batch_selected = False
    phase2_last_batch_penalty_total = 0.0
    phase2_last_optimizer_memory_reused = False
    phase2_last_optimizer_memory_source = "unavailable"
    phase2_last_shortlist_eval_records: list[dict[str, Any]] = []
    phase3_oracle_gradient_enabled = bool(phase3_oracle_gradient_config is not None)
    phase3_gradient_uncertainty_source = (
        "oracle_fd_stderr_v1" if phase3_oracle_gradient_enabled else "zero_default"
    )
    phase3_gradient_source_name = (
        "oracle_fd_v1" if phase3_oracle_gradient_enabled else "exact_commutator"
    )
    phase3_oracle_gradient_calls_total = 0
    phase3_oracle_backend_info: dict[str, Any] | None = None
    phase3_last_oracle_gradient_backend_info: dict[str, Any] | None = None
    phase3_oracle_inner_backend_info: dict[str, Any] | None = None
    phase3_last_candidate_gradient_scout: list[dict[str, Any]] = []
    phase3_last_max_gradient_stderr = 0.0
    phase3_oracle_gradient_raw_records_total = 0
    phase3_oracle_symmetry_diagnostic_calls_total = 0
    phase3_oracle_symmetry_diagnostic_raw_records_total = 0
    phase3_oracle_inner_objective_calls_total = 0
    phase3_oracle_inner_objective_raw_records_total = 0
    phase3_oracle_raw_transport: str | None = None
    phase3_oracle_raw_artifact_path: str | None = None
    phase3_oracle = None
    phase3_oracle_cleanup = None
    phase3_oracle_h_qop = None
    phase3_oracle_all_z_qop = None
    phase3_runtime_bindings_cache: dict[str, Any] | None = None
    build_runtime_layout_circuit_fn = None
    build_parameterized_ansatz_plan_fn = None
    phase3_oracle_num_qubits = int(round(math.log2(psi_ref.size)))
    phase3_oracle_plan_cache: dict[str, Any] = {}

    def _get_phase3_oracle_plan(layout_now: AnsatzParameterLayout) -> Any:
        if build_parameterized_ansatz_plan_fn is None:
            raise RuntimeError("phase3 oracle path is missing its parameterized ansatz plan builder.")
        cache_key = json.dumps(serialize_layout(layout_now), sort_keys=True, separators=(",", ":"))
        cached_plan = phase3_oracle_plan_cache.get(cache_key)
        if cached_plan is not None:
            return cached_plan
        plan_now = build_parameterized_ansatz_plan_fn(
            layout_now,
            nq=int(phase3_oracle_num_qubits),
            ref_state=np.asarray(psi_ref, dtype=complex),
        )
        phase3_oracle_plan_cache[cache_key] = plan_now
        return plan_now

    def _normalize_phase3_oracle_backend_info(
        *,
        oracle_obj: Any | None = None,
        raw_bundle: Any | None = None,
    ) -> dict[str, Any] | None:
        if raw_bundle is not None and phase3_oracle_gradient_config is not None:
            return _json_ready(
                {
                    "noise_mode": str(phase3_oracle_gradient_config.noise_mode),
                    "estimator_kind": "raw_measurement_oracle",
                    "backend_name": raw_bundle.backend_snapshot.get(
                        "backend_name", phase3_oracle_gradient_config.backend_name
                    ),
                    "using_fake_backend": bool(phase3_oracle_gradient_config.use_fake_backend),
                    "details": {
                        "execution_surface": "raw_measurement_v1",
                        "transport": str(raw_bundle.transport),
                        "raw_artifact_path": raw_bundle.raw_artifact_path,
                        "record_count": int(raw_bundle.estimate.record_count),
                        "group_count": int(raw_bundle.estimate.group_count),
                        "term_count": int(raw_bundle.estimate.term_count),
                        "reduction_mode": str(raw_bundle.estimate.reduction_mode),
                        "plan_digest": str(raw_bundle.plan_digest),
                        "structure_digest": str(raw_bundle.structure_digest),
                        "reference_state_digest": raw_bundle.reference_state_digest,
                        "compile_signatures_by_basis": dict(raw_bundle.compile_signatures_by_basis),
                        "backend_snapshot": dict(raw_bundle.backend_snapshot),
                        "transpile_seed": phase3_oracle_gradient_config.seed_transpiler,
                        "seed_transpiler": phase3_oracle_gradient_config.seed_transpiler,
                        "transpile_optimization_level": int(
                            phase3_oracle_gradient_config.transpile_optimization_level
                        ),
                    },
                }
            )
        backend_info_raw = getattr(oracle_obj, "backend_info", None) if oracle_obj is not None else None
        if backend_info_raw is not None:
            return _json_ready(getattr(backend_info_raw, "__dict__", backend_info_raw))
        if oracle_obj is not None and phase3_oracle_gradient_config is not None:
            return _json_ready(
                {
                    "noise_mode": str(phase3_oracle_gradient_config.noise_mode),
                    "estimator_kind": "raw_measurement_oracle",
                    "backend_name": phase3_oracle_gradient_config.backend_name,
                    "using_fake_backend": bool(phase3_oracle_gradient_config.use_fake_backend),
                    "details": {
                        "execution_surface": str(phase3_oracle_gradient_config.execution_surface),
                        "transport": getattr(oracle_obj, "transport", None),
                        "raw_artifact_path": phase3_oracle_gradient_config.raw_artifact_path,
                        "backend_snapshot": dict(getattr(oracle_obj, "backend_snapshot", {}) or {}),
                    },
                }
            )
        return None

    def _close_phase3_oracle_resource(oracle_obj: Any | None) -> None:
        if oracle_obj is None:
            return
        close_oracle = getattr(oracle_obj, "close", None)
        if callable(close_oracle):
            close_oracle()

    def _run_final_noise_audit(
        snapshot: FinalNoiseAuditSnapshot,
        config: FinalNoiseAuditConfig,
    ) -> dict[str, Any]:
        audit_cfg = _resolve_final_noise_audit_config(config)
        bindings = (
            phase3_runtime_bindings_cache
            if phase3_runtime_bindings_cache is not None
            else _phase3_oracle_runtime_bindings()
        )
        plan = bindings["build_parameterized_ansatz_plan"](
            snapshot.parameter_layout,
            nq=int(snapshot.num_qubits),
            ref_state=np.asarray(snapshot.reference_state, dtype=complex),
        )
        observable = bindings["pauli_poly_to_sparse_pauli_op"](snapshot.h_poly)
        theta_runtime = np.asarray(snapshot.theta_runtime, dtype=float)

        def _evaluate_variant(
            variant_cfg: FinalNoiseAuditConfig,
            *,
            audit_variant: str,
        ) -> dict[str, Any]:
            oracle_config = bindings["OracleConfig"](
                noise_mode=str(variant_cfg.noise_mode),
                shots=int(variant_cfg.shots),
                seed=int(variant_cfg.seed),
                seed_transpiler=variant_cfg.seed_transpiler,
                transpile_optimization_level=int(variant_cfg.transpile_optimization_level),
                oracle_repeats=int(variant_cfg.oracle_repeats),
                oracle_aggregate=str(variant_cfg.oracle_aggregate),
                backend_name=(
                    None
                    if variant_cfg.backend_name in {None, ""}
                    else str(variant_cfg.backend_name)
                ),
                use_fake_backend=bool(variant_cfg.use_fake_backend),
                allow_aer_fallback=True,
                aer_fallback_mode="sampler_shots",
                omp_shm_workaround=True,
                mitigation=dict(
                    _oracle_mitigation_payload_from_fields(
                        mitigation_mode=str(variant_cfg.mitigation_mode),
                        local_readout_strategy=variant_cfg.local_readout_strategy,
                        zne_scales=tuple(getattr(variant_cfg, "zne_scales", ()) or ()),
                        dd_sequence=getattr(variant_cfg, "dd_sequence", None),
                        local_gate_twirling=bool(
                            getattr(variant_cfg, "local_gate_twirling", False)
                        ),
                    )
                ),
                symmetry_mitigation={"mode": "off"},
                runtime_profile=str(variant_cfg.runtime_profile_name),
                runtime_session=str(variant_cfg.runtime_session_policy),
                execution_surface="expectation_v1",
            )
            validation_report = _validate_oracle_execution_request_via_bindings(bindings, oracle_config)
            normalized_request = (
                None
                if validation_report is None
                else dict(validation_report.get("normalized_request", {}) or {})
            )
            if (
                str(variant_cfg.noise_mode).strip().lower() == "backend_scheduled"
                and bool(variant_cfg.use_fake_backend)
            ):
                bindings["preflight_backend_scheduled_fake_backend_environment"](oracle_config)

            oracle_obj = bindings["ExpectationOracle"](oracle_config)
            close_oracle = getattr(oracle_obj, "close", None)
            try:
                if hasattr(oracle_obj, "evaluate_parameterized"):
                    estimate = oracle_obj.evaluate_parameterized(
                        plan=plan,
                        theta_runtime=theta_runtime,
                        observable=observable,
                        runtime_trace_context={
                            "route": "final_noise_audit_v1",
                            "audit_variant": str(audit_variant),
                            "ansatz_depth": int(snapshot.ansatz_depth),
                            "logical_parameter_count": int(snapshot.logical_parameter_count),
                            "runtime_parameter_count": int(snapshot.runtime_parameter_count),
                        },
                    )
                else:
                    circuit_obj = bindings["build_runtime_layout_circuit"](
                        snapshot.parameter_layout,
                        theta_runtime,
                        int(snapshot.num_qubits),
                        reference_state=np.asarray(snapshot.reference_state, dtype=complex),
                    )
                    try:
                        setattr(circuit_obj, "_final_noise_audit_route", "final_noise_audit_v1")
                        setattr(circuit_obj, "_final_noise_audit_variant", str(audit_variant))
                    except Exception:
                        pass
                    estimate = oracle_obj.evaluate(circuit_obj, observable)
                variant_energy = float(getattr(estimate, "mean", 0.0))
                exact_target_delta_e = float(
                    variant_energy - float(snapshot.exact_filtered_ground_energy)
                )
                exact_final_state_delta_e = float(
                    variant_energy - float(snapshot.exact_final_state_energy)
                )
                return {
                    "requested_config": dict(_final_noise_audit_config_payload(variant_cfg) or {}),
                    "normalized_request": normalized_request,
                    "result": {
                        "requested_estimate_energy": float(variant_energy),
                        "stderr": float(getattr(estimate, "stderr", 0.0) or 0.0),
                        "std": float(getattr(estimate, "std", 0.0) or 0.0),
                        "stdev": float(getattr(estimate, "stdev", 0.0) or 0.0),
                        "n_samples": int(getattr(estimate, "n_samples", 0) or 0),
                        "aggregate": str(getattr(estimate, "aggregate", variant_cfg.oracle_aggregate)),
                        "backend_info": _json_ready(
                            getattr(
                                getattr(oracle_obj, "backend_info", None),
                                "__dict__",
                                getattr(oracle_obj, "backend_info", None),
                            )
                        ),
                    },
                    "deltas": {
                        "exact_target_delta_e": float(exact_target_delta_e),
                        "exact_target_abs_error": float(abs(exact_target_delta_e)),
                        "exact_final_state_delta_e": float(exact_final_state_delta_e),
                        "exact_final_state_abs_error": float(abs(exact_final_state_delta_e)),
                    },
                }
            finally:
                if callable(close_oracle):
                    close_oracle()

        requested_eval = _evaluate_variant(audit_cfg, audit_variant="requested")
        output = {
            "status": "completed",
            "strict": bool(audit_cfg.strict),
            "requested_config": dict(requested_eval.get("requested_config", {})),
            "normalized_request": dict(requested_eval.get("normalized_request", {}) or {}),
            "reference": {
                "primary_metric_name": "exact_target_abs_error",
                "exact_filtered_ground_energy": float(snapshot.exact_filtered_ground_energy),
                "exact_final_state_energy": float(snapshot.exact_final_state_energy),
            },
            "snapshot": {
                "ansatz_depth": int(snapshot.ansatz_depth),
                "runtime_parameter_count": int(snapshot.runtime_parameter_count),
                "logical_parameter_count": int(snapshot.logical_parameter_count),
                "operator_labels": [str(x) for x in snapshot.operator_labels],
                "parameterization": serialize_layout(snapshot.parameter_layout),
                "theta_runtime": [float(x) for x in snapshot.theta_runtime],
                "theta_logical": [float(x) for x in snapshot.theta_logical],
            },
            "result": dict(requested_eval.get("result", {})),
            "deltas": dict(requested_eval.get("deltas", {})),
        }

        def _build_unmitigated_baseline_config(
            source_cfg: FinalNoiseAuditConfig,
        ) -> FinalNoiseAuditConfig:
            return _resolve_final_noise_audit_config(
                FinalNoiseAuditConfig(
                    noise_mode=str(source_cfg.noise_mode),
                    shots=int(source_cfg.shots),
                    oracle_repeats=int(source_cfg.oracle_repeats),
                    oracle_aggregate=str(source_cfg.oracle_aggregate),
                    backend_name=(
                        None
                        if source_cfg.backend_name in {None, ""}
                        else str(source_cfg.backend_name)
                    ),
                    use_fake_backend=bool(source_cfg.use_fake_backend),
                    seed=int(source_cfg.seed),
                    mitigation_mode="none",
                    local_readout_strategy=None,
                    zne_scales=(),
                    local_gate_twirling=False,
                    dd_sequence=None,
                    runtime_profile_name="legacy_runtime_v0",
                    runtime_session_policy=str(source_cfg.runtime_session_policy),
                    compare_unmitigated_baseline=False,
                    seed_transpiler=source_cfg.seed_transpiler,
                    transpile_optimization_level=int(source_cfg.transpile_optimization_level),
                    strict=bool(source_cfg.strict),
                )
            )

        baseline_requested = bool(audit_cfg.compare_unmitigated_baseline)
        if baseline_requested:
            baseline_cfg = _build_unmitigated_baseline_config(audit_cfg)
            baseline_requested_payload = dict(_final_noise_audit_config_payload(baseline_cfg) or {})
            requested_cfg_payload = dict(output.get("requested_config", {}) or {})
            requested_cfg_payload_cmp = dict(requested_cfg_payload)
            requested_cfg_payload_cmp["compare_unmitigated_baseline"] = False
            if baseline_requested_payload == requested_cfg_payload_cmp:
                output["unmitigated_baseline_comparison"] = {
                    "enabled": True,
                    "status": "skipped",
                    "reason": "requested_matches_unmitigated_baseline",
                    "baseline_requested_config": baseline_requested_payload,
                }
            else:
                try:
                    baseline_eval = _evaluate_variant(
                        baseline_cfg,
                        audit_variant="unmitigated_baseline",
                    )
                    requested_energy = float(output.get("result", {}).get("requested_estimate_energy", 0.0))
                    baseline_energy = float(
                        baseline_eval.get("result", {}).get("requested_estimate_energy", 0.0)
                    )
                    requested_exact_target_abs_error = float(
                        output.get("deltas", {}).get("exact_target_abs_error", 0.0)
                    )
                    baseline_exact_target_abs_error = float(
                        baseline_eval.get("deltas", {}).get("exact_target_abs_error", 0.0)
                    )
                    requested_exact_final_state_abs_error = float(
                        output.get("deltas", {}).get("exact_final_state_abs_error", 0.0)
                    )
                    baseline_exact_final_state_abs_error = float(
                        baseline_eval.get("deltas", {}).get("exact_final_state_abs_error", 0.0)
                    )
                    output["unmitigated_baseline_comparison"] = {
                        "enabled": True,
                        "status": "completed",
                        "baseline_requested_config": dict(
                            baseline_eval.get("requested_config", {})
                        ),
                        "baseline_normalized_request": dict(
                            baseline_eval.get("normalized_request", {}) or {}
                        ),
                        "baseline_result": dict(baseline_eval.get("result", {})),
                        "baseline_deltas": dict(baseline_eval.get("deltas", {})),
                        "comparison_metrics": {
                            "requested_minus_unmitigated_delta_e": float(
                                requested_energy - baseline_energy
                            ),
                            "requested_minus_unmitigated_abs_delta_e": float(
                                abs(requested_energy - baseline_energy)
                            ),
                            "exact_target_abs_error_improvement_vs_unmitigated": float(
                                baseline_exact_target_abs_error
                                - requested_exact_target_abs_error
                            ),
                            "exact_final_state_abs_error_improvement_vs_unmitigated": float(
                                baseline_exact_final_state_abs_error
                                - requested_exact_final_state_abs_error
                            ),
                        },
                    }
                except Exception as exc:
                    if bool(audit_cfg.strict):
                        raise
                    output["unmitigated_baseline_comparison"] = {
                        "enabled": True,
                        "status": "failed",
                        "reason": "evaluation_failed",
                        "baseline_requested_config": baseline_requested_payload,
                        "failure": {
                            "error_type": str(type(exc).__name__),
                            "error_message": str(exc),
                        },
                    }
        return output

    class _Phase3OracleCleanupGuard:
        def __init__(self, oracle_obj: Any | None) -> None:
            self._oracle_ref = weakref.ref(oracle_obj) if oracle_obj is not None else None
            self._closed = False

        def close(self) -> None:
            if self._closed:
                return
            self._closed = True
            oracle_obj = self._oracle_ref() if self._oracle_ref is not None else None
            _close_phase3_oracle_resource(oracle_obj)

        def __del__(self) -> None:
            try:
                self.close()
            except Exception:
                pass

    if phase3_oracle_gradient_enabled:
        bindings = _phase3_oracle_runtime_bindings()
        phase3_runtime_bindings_cache = dict(bindings)
        build_parameterized_ansatz_plan_fn = bindings["build_parameterized_ansatz_plan"]
        oracle_config = bindings["OracleConfig"](
            noise_mode=str(phase3_oracle_gradient_config.noise_mode),
            shots=int(phase3_oracle_gradient_config.shots),
            seed=int(phase3_oracle_gradient_config.seed),
            seed_transpiler=phase3_oracle_gradient_config.seed_transpiler,
            transpile_optimization_level=int(phase3_oracle_gradient_config.transpile_optimization_level),
            oracle_repeats=int(phase3_oracle_gradient_config.oracle_repeats),
            oracle_aggregate=str(phase3_oracle_gradient_config.oracle_aggregate),
            backend_name=(
                None
                if phase3_oracle_gradient_config.backend_name in {None, ""}
                else str(phase3_oracle_gradient_config.backend_name)
            ),
            use_fake_backend=bool(phase3_oracle_gradient_config.use_fake_backend),
            allow_aer_fallback=True,
            aer_fallback_mode="sampler_shots",
            omp_shm_workaround=True,
            mitigation=dict(_phase3_oracle_mitigation_payload(phase3_oracle_gradient_config)),
            symmetry_mitigation={"mode": "off"},
            execution_surface=str(phase3_oracle_gradient_config.execution_surface),
            raw_transport=str(phase3_oracle_gradient_config.raw_transport),
            raw_store_memory=bool(phase3_oracle_gradient_config.raw_store_memory),
            raw_artifact_path=phase3_oracle_gradient_config.raw_artifact_path,
        )
        _validate_oracle_execution_request_via_bindings(bindings, oracle_config)
        if (
            str(phase3_oracle_gradient_config.noise_mode).strip().lower() == "backend_scheduled"
            and bool(phase3_oracle_gradient_config.use_fake_backend)
        ):
            try:
                bindings["preflight_backend_scheduled_fake_backend_environment"](oracle_config)
                _ai_log(
                    "hardcoded_adapt_phase3_oracle_backend_scheduled_preflight_ok",
                    backend_name=oracle_config.backend_name,
                    execution_surface=str(oracle_config.execution_surface),
                )
            except Exception as exc:
                _ai_log(
                    "hardcoded_adapt_phase3_oracle_backend_scheduled_preflight_failed",
                    backend_name=oracle_config.backend_name,
                    execution_surface=str(oracle_config.execution_surface),
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
        if str(phase3_oracle_gradient_config.execution_surface) == "raw_measurement_v1":
            if str(phase3_oracle_gradient_config.noise_mode).strip().lower() == "runtime":
                oracle_config = bindings["normalize_sampler_raw_runtime_config"](oracle_config)
            phase3_oracle = bindings["RawMeasurementOracle"](oracle_config)
            phase3_oracle_all_z_qop = bindings["all_z_full_register_qop"](int(phase3_oracle_num_qubits))
            phase3_oracle_raw_transport = str(
                getattr(phase3_oracle, "transport", phase3_oracle_gradient_config.raw_transport)
            )
            phase3_oracle_raw_artifact_path = phase3_oracle_gradient_config.raw_artifact_path
            phase3_oracle_backend_info = _normalize_phase3_oracle_backend_info(oracle_obj=phase3_oracle)
        else:
            phase3_oracle = bindings["ExpectationOracle"](oracle_config)
            build_runtime_layout_circuit_fn = bindings["build_runtime_layout_circuit"]
            phase3_oracle_backend_info = _normalize_phase3_oracle_backend_info(oracle_obj=phase3_oracle)
        phase3_oracle_cleanup = _Phase3OracleCleanupGuard(phase3_oracle)
        phase3_oracle_h_qop = bindings["pauli_poly_to_sparse_pauli_op"](h_poly)
        if bool(finite_angle_fallback):
            _ai_log(
                "hardcoded_adapt_phase3_oracle_finite_angle_disabled",
                reason="oracle_gradient_mode_selection_only",
            )
        if bool(phase1_prune_enabled) and (not phase3_oracle_inner_objective_enabled):
            _ai_log(
                "hardcoded_adapt_phase3_oracle_prune_disabled",
                reason="oracle_gradient_mode_selection_only",
            )
        finite_angle_fallback = False
        if not phase3_oracle_inner_objective_enabled:
            phase1_prune_enabled = False

    try:
        def _phase3_oracle_gradient_scout(
            *,
            selected_ops_now: list[AnsatzTerm],
            theta_now: np.ndarray,
            append_position_now: int,
            available_indices_now: Sequence[int],
        ) -> tuple[np.ndarray, np.ndarray, dict[str, float], list[dict[str, Any]], int]:
            nonlocal phase3_last_oracle_gradient_backend_info
            nonlocal phase3_oracle_gradient_raw_records_total
            nonlocal phase3_oracle_symmetry_diagnostic_calls_total
            nonlocal phase3_oracle_symmetry_diagnostic_raw_records_total
            nonlocal phase3_oracle_raw_transport
            nonlocal phase3_oracle_raw_artifact_path
            if (
                not phase3_oracle_gradient_enabled
                or phase3_oracle is None
                or phase3_oracle_h_qop is None
                or phase3_oracle_gradient_config is None
            ):
                raise RuntimeError("phase3 oracle gradient scout requested without an active oracle session.")
            if (
                str(phase3_oracle_gradient_config.execution_surface) == "expectation_v1"
                and build_runtime_layout_circuit_fn is None
            ):
                raise RuntimeError("phase3 expectation oracle path is missing its circuit builder.")
            if (
                str(phase3_oracle_gradient_config.execution_surface) == "raw_measurement_v1"
                and build_parameterized_ansatz_plan_fn is None
            ):
                raise RuntimeError("phase3 raw oracle path is missing its plan builder.")
            gradients_local = np.zeros(len(pool), dtype=float)
            grad_magnitudes_local = np.zeros(len(pool), dtype=float)
            sigma_by_label_local: dict[str, float] = {}
            scout_rows_local: list[dict[str, Any]] = []
            oracle_calls_local = 0
            grad_step_local = float(phase3_oracle_gradient_config.gradient_step)
            available_idx_sorted = [int(i) for i in sorted(int(i) for i in available_indices_now)]
            _ai_log(
                "hardcoded_adapt_phase3_oracle_scout_start",
                depth=int(depth + 1),
                available_count=int(len(available_idx_sorted)),
                append_position=int(append_position_now),
                noise_mode=str(phase3_oracle_gradient_config.noise_mode),
                execution_surface=str(phase3_oracle_gradient_config.execution_surface),
                raw_transport=(
                    None
                    if str(phase3_oracle_gradient_config.execution_surface) != "raw_measurement_v1"
                    else str(phase3_oracle_gradient_config.raw_transport)
                ),
                shots=int(phase3_oracle_gradient_config.shots),
                oracle_repeats=int(phase3_oracle_gradient_config.oracle_repeats),
                mitigation_mode=str(phase3_oracle_gradient_config.mitigation_mode),
                local_readout_strategy=(
                    None
                    if phase3_oracle_gradient_config.local_readout_strategy in {None, ""}
                    else str(phase3_oracle_gradient_config.local_readout_strategy)
                ),
            )

            def _measure_raw_probe_bundle(
                *,
                plan_probe: Any,
                theta_probe: np.ndarray,
                observable: Any,
                observable_family: str,
                probe_name: str,
                candidate_pool_index: int,
                candidate_label_value: str,
                append_position_value: int,
                extra_semantic_tags: Mapping[str, Any] | None = None,
            ) -> tuple[Any, int]:
                eval_t0 = time.perf_counter()
                _ai_log(
                    "hardcoded_adapt_phase3_oracle_eval_start",
                    depth=int(depth + 1),
                    candidate_pool_index=int(candidate_pool_index),
                    candidate_label=str(candidate_label_value),
                    probe_sign=str(probe_name),
                    observable_family=str(observable_family),
                )
                try:
                    bundle = phase3_oracle.measure_observable(
                        plan=plan_probe,
                        theta_runtime=np.asarray(theta_probe, dtype=float),
                        observable=observable,
                        observable_family=str(observable_family),
                        semantic_tags={
                            "route": "adapt_phase3_oracle_gradient",
                            "depth": int(depth + 1),
                            "candidate_pool_index": int(candidate_pool_index),
                            "candidate_label": str(candidate_label_value),
                            "append_position": int(append_position_value),
                            "probe_sign": str(probe_name),
                            "oracle_scope": str(phase3_oracle_gradient_config.scope),
                            **{str(k): v for k, v in dict(extra_semantic_tags or {}).items()},
                        },
                    )
                except Exception as exc:
                    _ai_log(
                        "hardcoded_adapt_phase3_oracle_eval_error",
                        depth=int(depth + 1),
                        candidate_pool_index=int(candidate_pool_index),
                        candidate_label=str(candidate_label_value),
                        probe_sign=str(probe_name),
                        observable_family=str(observable_family),
                        elapsed_s=float(time.perf_counter() - eval_t0),
                        error_type=str(type(exc).__name__),
                        error_repr=repr(exc),
                    )
                    raise
                bundle_record_count = int(
                    getattr(
                        getattr(bundle, "estimate", None),
                        "record_count",
                        len(getattr(bundle, "records", ()) or ()),
                    )
                )
                _ai_log(
                    "hardcoded_adapt_phase3_oracle_eval_done",
                    depth=int(depth + 1),
                    candidate_pool_index=int(candidate_pool_index),
                    candidate_label=str(candidate_label_value),
                    probe_sign=str(probe_name),
                    observable_family=str(observable_family),
                    elapsed_s=float(time.perf_counter() - eval_t0),
                    stderr=float(getattr(bundle.estimate, "stderr", 0.0) or 0.0),
                    std=float(getattr(bundle.estimate, "std", 0.0) or 0.0),
                    n_samples=int(getattr(bundle.estimate, "n_samples", 0) or 0),
                    aggregate=str(
                        getattr(bundle.estimate, "aggregate", phase3_oracle_gradient_config.oracle_aggregate)
                    ),
                    record_count=int(bundle_record_count),
                    transport=str(bundle.transport),
                )
                return bundle, int(bundle_record_count)

            for idx in available_idx_sorted:
                candidate_term = pool[int(idx)]
                candidate_label = str(candidate_term.label)
                ops_plus, theta_plus = _splice_candidate_at_position(
                    ops=list(selected_ops_now),
                    theta=np.asarray(theta_now, dtype=float),
                    op=candidate_term,
                    position_id=int(append_position_now),
                    init_theta=float(grad_step_local),
                )
                ops_minus, theta_minus = _splice_candidate_at_position(
                    ops=list(selected_ops_now),
                    theta=np.asarray(theta_now, dtype=float),
                    op=candidate_term,
                    position_id=int(append_position_now),
                    init_theta=-float(grad_step_local),
                )
                raw_summary_local: dict[str, Any] | None = None
                if str(phase3_oracle_gradient_config.execution_surface) == "raw_measurement_v1":
                    if phase3_oracle_all_z_qop is None:
                        raise RuntimeError("phase3 raw oracle path is missing its all-Z diagnostic observable.")
                    layout_plus = _build_selected_layout(ops_plus)
                    layout_minus = _build_selected_layout(ops_minus)
                    if serialize_layout(layout_plus) != serialize_layout(layout_minus):
                        raise RuntimeError("phase3 raw finite-difference probe structure mismatch")
                    plan_probe = build_parameterized_ansatz_plan_fn(
                        layout_plus,
                        nq=int(phase3_oracle_num_qubits),
                        ref_state=np.asarray(psi_ref, dtype=complex),
                    )
                    raw_bundles: dict[str, Any] = {}
                    raw_record_counts: dict[str, int] = {}
                    symmetry_diagnostic_local: dict[str, Any] = {}
                    for probe_name, theta_probe in (("plus", theta_plus), ("minus", theta_minus)):
                        bundle, bundle_record_count = _measure_raw_probe_bundle(
                            plan_probe=plan_probe,
                            theta_probe=np.asarray(theta_probe, dtype=float),
                            observable=phase3_oracle_h_qop,
                            observable_family="adapt_phase3_oracle_gradient",
                            probe_name=str(probe_name),
                            candidate_pool_index=int(idx),
                            candidate_label_value=str(candidate_label),
                            append_position_value=int(append_position_now),
                        )
                        oracle_calls_local += 1
                        raw_bundles[str(probe_name)] = bundle
                        raw_record_counts[str(probe_name)] = int(bundle_record_count)
                        phase3_oracle_gradient_raw_records_total += int(bundle_record_count)
                        phase3_oracle_raw_transport = str(bundle.transport)
                        if bundle.raw_artifact_path not in {None, ""}:
                            phase3_oracle_raw_artifact_path = str(bundle.raw_artifact_path)
                        phase3_last_oracle_gradient_backend_info = _normalize_phase3_oracle_backend_info(raw_bundle=bundle)

                        diagnostic_payload = {
                            "available": False,
                            "reason": None,
                            "observable_family": "adapt_phase3_oracle_symmetry_diagnostic",
                            "evaluation_id": None,
                            "transport": None,
                            "raw_artifact_path": phase3_oracle_raw_artifact_path,
                            "record_count_total": 0,
                            "group_count": None,
                            "term_count": None,
                            "compile_signatures_by_basis": {},
                            "summary": None,
                            "error_type": None,
                            "error_message": None,
                        }
                        try:
                            diagnostic_bundle, diagnostic_record_count = _measure_raw_probe_bundle(
                                plan_probe=plan_probe,
                                theta_probe=np.asarray(theta_probe, dtype=float),
                                observable=phase3_oracle_all_z_qop,
                                observable_family="adapt_phase3_oracle_symmetry_diagnostic",
                                probe_name=str(probe_name),
                                candidate_pool_index=int(idx),
                                candidate_label_value=str(candidate_label),
                                append_position_value=int(append_position_now),
                                extra_semantic_tags={
                                    "diagnostic_kind": "all_z_full_register_v1",
                                    "symmetry_num_sites": int(num_sites),
                                    "symmetry_ordering": str(ordering),
                                    "symmetry_sector_n_up": int(num_particles[0]),
                                    "symmetry_sector_n_dn": int(num_particles[1]),
                                },
                            )
                            phase3_oracle_symmetry_diagnostic_calls_total += 1
                            phase3_oracle_symmetry_diagnostic_raw_records_total += int(diagnostic_record_count)
                            if diagnostic_bundle.raw_artifact_path not in {None, ""}:
                                phase3_oracle_raw_artifact_path = str(diagnostic_bundle.raw_artifact_path)
                            diagnostic_payload.update(
                                {
                                    "available": True,
                                    "reason": None,
                                    "observable_family": str(diagnostic_bundle.observable_family),
                                    "evaluation_id": str(diagnostic_bundle.evaluation_id),
                                    "transport": str(diagnostic_bundle.transport),
                                    "raw_artifact_path": diagnostic_bundle.raw_artifact_path,
                                    "record_count_total": int(diagnostic_record_count),
                                    "group_count": int(diagnostic_bundle.estimate.group_count),
                                    "term_count": int(diagnostic_bundle.estimate.term_count),
                                    "compile_signatures_by_basis": dict(
                                        diagnostic_bundle.compile_signatures_by_basis
                                    ),
                                }
                            )
                            try:
                                diagnostic_payload["summary"] = _json_ready(
                                    bindings["summarize_hh_full_register_z_records"](
                                        getattr(diagnostic_bundle, "records", ()),
                                        num_sites=int(num_sites),
                                        ordering=str(ordering),
                                        sector_n_up=int(num_particles[0]),
                                        sector_n_dn=int(num_particles[1]),
                                        expected_repeat_count=int(
                                            phase3_oracle_gradient_config.oracle_repeats
                                        ),
                                    )
                                )
                            except Exception as exc:
                                diagnostic_payload.update(
                                    {
                                        "available": False,
                                        "reason": "summary_failed",
                                        "summary": None,
                                        "error_type": str(type(exc).__name__),
                                        "error_message": str(exc),
                                    }
                                )
                                _ai_log(
                                    "hardcoded_adapt_phase3_oracle_symmetry_diagnostic_unavailable",
                                    depth=int(depth + 1),
                                    candidate_pool_index=int(idx),
                                    candidate_label=str(candidate_label),
                                    probe_sign=str(probe_name),
                                    reason="summary_failed",
                                    error_type=str(type(exc).__name__),
                                    error_repr=repr(exc),
                                )
                        except Exception as exc:
                            diagnostic_payload.update(
                                {
                                    "available": False,
                                    "reason": "measurement_failed",
                                    "summary": None,
                                    "error_type": str(type(exc).__name__),
                                    "error_message": str(exc),
                                }
                            )
                            _ai_log(
                                "hardcoded_adapt_phase3_oracle_symmetry_diagnostic_unavailable",
                                depth=int(depth + 1),
                                candidate_pool_index=int(idx),
                                candidate_label=str(candidate_label),
                                probe_sign=str(probe_name),
                                reason="measurement_failed",
                                error_type=str(type(exc).__name__),
                                error_repr=repr(exc),
                            )
                        symmetry_diagnostic_local[str(probe_name)] = dict(diagnostic_payload)
                    bundle_plus = raw_bundles["plus"]
                    bundle_minus = raw_bundles["minus"]
                    e_plus = bundle_plus.estimate
                    e_minus = bundle_minus.estimate
                    raw_summary_local = {
                        "transport": str(bundle_plus.transport),
                        "record_count_total": int(raw_record_counts.get("plus", 0) + raw_record_counts.get("minus", 0)),
                        "group_count": int(bundle_plus.estimate.group_count),
                        "reduction_mode_plus": str(bundle_plus.estimate.reduction_mode),
                        "reduction_mode_minus": str(bundle_minus.estimate.reduction_mode),
                        "evaluation_id_plus": str(bundle_plus.evaluation_id),
                        "evaluation_id_minus": str(bundle_minus.evaluation_id),
                        "raw_artifact_path": (
                            bundle_plus.raw_artifact_path
                            if bundle_plus.raw_artifact_path not in {None, ""}
                            else bundle_minus.raw_artifact_path
                        ),
                        "symmetry_diagnostic": dict(symmetry_diagnostic_local),
                    }
                else:
                    layout_plus = _build_selected_layout(ops_plus)
                    layout_minus = _build_selected_layout(ops_minus)
                    if serialize_layout(layout_plus) != serialize_layout(layout_minus):
                        raise RuntimeError("phase3 expectation finite-difference probe structure mismatch")
                    oracle_estimates: dict[str, Any] = {}
                    if hasattr(phase3_oracle, "evaluate_parameterized"):
                        plan_probe = _get_phase3_oracle_plan(layout_plus)
                        probe_iter = (("plus", np.asarray(theta_plus, dtype=float)), ("minus", np.asarray(theta_minus, dtype=float)))
                        for probe_name, theta_probe in probe_iter:
                            eval_t0 = time.perf_counter()
                            _ai_log(
                                "hardcoded_adapt_phase3_oracle_eval_start",
                                depth=int(depth + 1),
                                candidate_pool_index=int(idx),
                                candidate_label=str(candidate_label),
                                probe_sign=str(probe_name),
                            )
                            try:
                                estimate = phase3_oracle.evaluate_parameterized(
                                    plan=plan_probe,
                                    theta_runtime=theta_probe,
                                    observable=phase3_oracle_h_qop,
                                    runtime_trace_context={
                                        "route": "adapt_phase3_oracle_gradient",
                                        "candidate_pool_index": int(idx),
                                        "candidate_label": str(candidate_label),
                                        "probe_sign": str(probe_name),
                                        "depth": int(depth + 1),
                                    },
                                )
                            except Exception as exc:
                                _ai_log(
                                    "hardcoded_adapt_phase3_oracle_eval_error",
                                    depth=int(depth + 1),
                                    candidate_pool_index=int(idx),
                                    candidate_label=str(candidate_label),
                                    probe_sign=str(probe_name),
                                    elapsed_s=float(time.perf_counter() - eval_t0),
                                    error_type=str(type(exc).__name__),
                                    error_repr=repr(exc),
                                )
                                raise
                            oracle_calls_local += 1
                            oracle_estimates[str(probe_name)] = estimate
                            phase3_last_oracle_gradient_backend_info = _normalize_phase3_oracle_backend_info(
                                oracle_obj=phase3_oracle
                            )
                            _ai_log(
                                "hardcoded_adapt_phase3_oracle_eval_done",
                                depth=int(depth + 1),
                                candidate_pool_index=int(idx),
                                candidate_label=str(candidate_label),
                                probe_sign=str(probe_name),
                                elapsed_s=float(time.perf_counter() - eval_t0),
                                stderr=float(getattr(estimate, "stderr", 0.0) or 0.0),
                                std=float(getattr(estimate, "std", 0.0) or 0.0),
                                n_samples=int(getattr(estimate, "n_samples", 0) or 0),
                                aggregate=str(
                                    getattr(estimate, "aggregate", phase3_oracle_gradient_config.oracle_aggregate)
                                ),
                            )
                    else:
                        circuit_plus = build_runtime_layout_circuit_fn(
                            layout_plus,
                            theta_plus,
                            int(phase3_oracle_num_qubits),
                            reference_state=np.asarray(psi_ref, dtype=complex),
                        )
                        circuit_minus = build_runtime_layout_circuit_fn(
                            layout_minus,
                            theta_minus,
                            int(phase3_oracle_num_qubits),
                            reference_state=np.asarray(psi_ref, dtype=complex),
                        )
                        for circuit_obj, sign in ((circuit_plus, 1.0), (circuit_minus, -1.0)):
                            try:
                                setattr(circuit_obj, "_phase3_candidate_label", str(candidate_label))
                                setattr(circuit_obj, "_phase3_probe_sign", float(sign))
                            except Exception:
                                pass
                        for probe_name, circuit_obj in (("plus", circuit_plus), ("minus", circuit_minus)):
                            eval_t0 = time.perf_counter()
                            _ai_log(
                                "hardcoded_adapt_phase3_oracle_eval_start",
                                depth=int(depth + 1),
                                candidate_pool_index=int(idx),
                                candidate_label=str(candidate_label),
                                probe_sign=str(probe_name),
                            )
                            try:
                                estimate = phase3_oracle.evaluate(circuit_obj, phase3_oracle_h_qop)
                            except Exception as exc:
                                _ai_log(
                                    "hardcoded_adapt_phase3_oracle_eval_error",
                                    depth=int(depth + 1),
                                    candidate_pool_index=int(idx),
                                    candidate_label=str(candidate_label),
                                    probe_sign=str(probe_name),
                                    elapsed_s=float(time.perf_counter() - eval_t0),
                                    error_type=str(type(exc).__name__),
                                    error_repr=repr(exc),
                                )
                                raise
                            oracle_calls_local += 1
                            oracle_estimates[str(probe_name)] = estimate
                            phase3_last_oracle_gradient_backend_info = _normalize_phase3_oracle_backend_info(
                                oracle_obj=phase3_oracle
                            )
                            _ai_log(
                                "hardcoded_adapt_phase3_oracle_eval_done",
                                depth=int(depth + 1),
                                candidate_pool_index=int(idx),
                                candidate_label=str(candidate_label),
                                probe_sign=str(probe_name),
                                elapsed_s=float(time.perf_counter() - eval_t0),
                                stderr=float(getattr(estimate, "stderr", 0.0) or 0.0),
                                std=float(getattr(estimate, "std", 0.0) or 0.0),
                                n_samples=int(getattr(estimate, "n_samples", 0) or 0),
                                aggregate=str(
                                    getattr(estimate, "aggregate", phase3_oracle_gradient_config.oracle_aggregate)
                                ),
                            )
                    e_plus = oracle_estimates["plus"]
                    e_minus = oracle_estimates["minus"]
                gradient_signed = float((float(e_plus.mean) - float(e_minus.mean)) / (2.0 * grad_step_local))
                sigma_hat_local = float(
                    _oracle_fd_gradient_stderr(e_plus, e_minus, grad_step=grad_step_local)
                )
                g_abs_local = float(abs(gradient_signed))
                g_lcb_local = max(
                    float(g_abs_local) - float(phase1_score_cfg.z_alpha) * float(sigma_hat_local),
                    0.0,
                )
                gradients_local[int(idx)] = float(gradient_signed)
                grad_magnitudes_local[int(idx)] = float(g_abs_local)
                sigma_by_label_local[str(candidate_label)] = float(sigma_hat_local)
                scout_rows_local.append(
                    {
                        "candidate_pool_index": int(idx),
                        "candidate_label": str(candidate_label),
                        "gradient_signed": float(gradient_signed),
                        "gradient_abs": float(g_abs_local),
                        "gradient_stderr": float(sigma_hat_local),
                        "sigma_hat": float(sigma_hat_local),
                        "g_lcb": float(g_lcb_local),
                        "oracle_samples_plus": int(getattr(e_plus, "n_samples", 0)),
                        "oracle_samples_minus": int(getattr(e_minus, "n_samples", 0)),
                        "oracle_aggregate": str(
                            getattr(e_plus, "aggregate", phase3_oracle_gradient_config.oracle_aggregate)
                        ),
                        "raw_summary": (
                            dict(raw_summary_local) if isinstance(raw_summary_local, Mapping) else None
                        ),
                        "selected_for_optimization": False,
                    }
                )
            _ai_log(
                "hardcoded_adapt_phase3_oracle_scout_done",
                depth=int(depth + 1),
                available_count=int(len(available_idx_sorted)),
                oracle_calls_total=int(oracle_calls_local),
                max_gradient_stderr=float(
                    max((float(row.get("gradient_stderr", 0.0)) for row in scout_rows_local), default=0.0)
                ),
                raw_records_total=int(phase3_oracle_gradient_raw_records_total),
            )
            return gradients_local, grad_magnitudes_local, sigma_by_label_local, scout_rows_local, int(oracle_calls_local)

        def _evaluate_selected_energy_objective(
            *,
            ops_now: Sequence[AnsatzTerm],
            theta_now: np.ndarray,
            executor_now: CompiledAnsatzExecutor | None,
            parameter_layout_now: AnsatzParameterLayout | None,
            objective_stage: str,
            depth_marker: int | None = None,
        ) -> float:
            nonlocal phase3_oracle_inner_backend_info
            nonlocal phase3_oracle_inner_objective_calls_total
            nonlocal phase3_oracle_inner_objective_raw_records_total
            theta_eval = np.asarray(theta_now, dtype=float)
            layout_eval = (
                parameter_layout_now
                if parameter_layout_now is not None
                else _build_selected_layout(list(ops_now))
            )
            if phase3_oracle_inner_objective_enabled:
                if (
                    phase3_oracle is None
                    or phase3_oracle_h_qop is None
                    or phase3_oracle_gradient_config is None
                ):
                    raise RuntimeError(
                        "phase3 noisy inner objective requested without an active oracle session."
                    )
                eval_t0 = time.perf_counter()
                execution_surface = (
                    str(phase3_oracle_gradient_config.execution_surface).strip().lower()
                    or "expectation_v1"
                )
                _ai_log(
                    "hardcoded_adapt_phase3_oracle_inner_eval_start",
                    stage=str(objective_stage),
                    depth=(None if depth_marker is None else int(depth_marker)),
                    execution_surface=str(execution_surface),
                    theta_runtime_count=int(theta_eval.size),
                    operator_count=int(len(list(ops_now))),
                )
                bundle_record_count = 0
                transport = None
                try:
                    if execution_surface == "raw_measurement_v1":
                        if build_parameterized_ansatz_plan_fn is None:
                            raise RuntimeError(
                                "phase3 noisy inner raw objective requested without an active raw oracle session."
                            )
                        bundle = phase3_oracle.measure_observable(
                            plan=_get_phase3_oracle_plan(layout_eval),
                            theta_runtime=theta_eval,
                            observable=phase3_oracle_h_qop,
                            observable_family="adapt_phase3_oracle_inner_objective",
                            semantic_tags={
                                "route": "adapt_phase3_oracle_inner_objective",
                                "objective_stage": str(objective_stage),
                                "depth": (
                                    None if depth_marker is None else int(depth_marker)
                                ),
                                "operator_count": int(len(list(ops_now))),
                            },
                        )
                        estimate = bundle.estimate
                        bundle_record_count = int(
                            getattr(
                                bundle.estimate,
                                "record_count",
                                len(getattr(bundle, "records", ()) or ()),
                            )
                        )
                        phase3_oracle_inner_objective_raw_records_total += int(
                            bundle_record_count
                        )
                        phase3_oracle_inner_backend_info = _normalize_phase3_oracle_backend_info(
                            raw_bundle=bundle
                        )
                        transport = str(
                            getattr(bundle, "transport", phase3_oracle_raw_transport)
                        )
                    else:
                        if hasattr(phase3_oracle, "evaluate_parameterized"):
                            estimate = phase3_oracle.evaluate_parameterized(
                                plan=_get_phase3_oracle_plan(layout_eval),
                                theta_runtime=theta_eval,
                                observable=phase3_oracle_h_qop,
                                runtime_trace_context={
                                    "route": "adapt_phase3_oracle_inner_objective",
                                    "objective_stage": str(objective_stage),
                                    "depth": (
                                        None if depth_marker is None else int(depth_marker)
                                    ),
                                    "operator_count": int(len(list(ops_now))),
                                },
                            )
                        else:
                            if build_runtime_layout_circuit_fn is None:
                                raise RuntimeError(
                                    "phase3 noisy inner expectation objective requested without an active expectation oracle session."
                                )
                            circuit_obj = build_runtime_layout_circuit_fn(
                                layout_eval,
                                theta_eval,
                                int(phase3_oracle_num_qubits),
                                reference_state=np.asarray(psi_ref, dtype=complex),
                            )
                            try:
                                setattr(
                                    circuit_obj,
                                    "_phase3_objective_stage",
                                    str(objective_stage),
                                )
                                setattr(
                                    circuit_obj,
                                    "_phase3_objective_depth",
                                    int(depth_marker or 0),
                                )
                            except Exception:
                                pass
                            estimate = phase3_oracle.evaluate(
                                circuit_obj,
                                phase3_oracle_h_qop,
                            )
                        phase3_oracle_inner_backend_info = _normalize_phase3_oracle_backend_info(
                            oracle_obj=phase3_oracle
                        )
                except Exception as exc:
                    _ai_log(
                        "hardcoded_adapt_phase3_oracle_inner_eval_error",
                        stage=str(objective_stage),
                        depth=(None if depth_marker is None else int(depth_marker)),
                        execution_surface=str(execution_surface),
                        elapsed_s=float(time.perf_counter() - eval_t0),
                        error_type=str(type(exc).__name__),
                        error_repr=repr(exc),
                    )
                    raise
                phase3_oracle_inner_objective_calls_total += 1
                _ai_log(
                    "hardcoded_adapt_phase3_oracle_inner_eval_done",
                    stage=str(objective_stage),
                    depth=(None if depth_marker is None else int(depth_marker)),
                    execution_surface=str(execution_surface),
                    elapsed_s=float(time.perf_counter() - eval_t0),
                    mean=float(getattr(estimate, "mean", 0.0) or 0.0),
                    stderr=float(getattr(estimate, "stderr", 0.0) or 0.0),
                    std=float(getattr(estimate, "std", 0.0) or 0.0),
                    n_samples=int(getattr(estimate, "n_samples", 0) or 0),
                    record_count=int(bundle_record_count),
                    transport=(None if transport in {None, ""} else str(transport)),
                    aggregate=str(
                        getattr(
                            estimate,
                            "aggregate",
                            phase3_oracle_gradient_config.oracle_aggregate,
                        )
                    ),
                )
                return float(estimate.mean)
            if executor_now is not None:
                psi_obj = _prepare_selected_state(
                    ops_now=list(ops_now),
                    theta_now=theta_eval,
                    executor_now=executor_now,
                    parameter_layout_now=layout_eval,
                )
                energy_obj, _ = energy_via_one_apply(psi_obj, h_compiled)
                energy_exact = float(energy_obj)
            else:
                energy_exact = float(
                    _adapt_energy_fn(
                        h_poly,
                        psi_ref,
                        list(ops_now),
                        theta_eval,
                        h_compiled=h_compiled,
                        parameter_layout=parameter_layout_now,
                    )
                )
            if adapt_analytic_noise_enabled:
                return _add_adapt_analytic_noise(energy_exact)
            return energy_exact

        energy_current = _evaluate_selected_energy_objective(
            ops_now=list(selected_ops),
            theta_now=np.asarray(theta, dtype=float),
            executor_now=selected_executor,
            parameter_layout_now=selected_layout,
            objective_stage="initial_state",
            depth_marker=0,
        )
        adapt_ref_import: dict[str, Any] | None = None
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
                seed_layout = _build_selected_layout(seed_ops)
                theta_seed0 = np.zeros(int(seed_layout.runtime_parameter_count), dtype=float)
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
                    seed_energy_val = _evaluate_selected_energy_objective(
                        ops_now=seed_ops,
                        theta_now=np.asarray(x, dtype=float),
                        executor_now=seed_executor,
                        parameter_layout_now=seed_layout,
                        objective_stage="hh_seed_preopt",
                        depth_marker=0,
                    )
                    if adapt_inner_optimizer_key in {"COBYLA", "POWELL"}:
                        seed_cobyla_nfev_so_far += 1
                        if seed_energy_val < seed_cobyla_best_fun:
                            seed_cobyla_best_fun = float(seed_energy_val)
                        now = time.perf_counter()
                        if (now - seed_cobyla_last_hb_t) >= float(adapt_spsa_progress_every_s):
                            _ai_log(
                                "hardcoded_adapt_scipy_heartbeat",
                                stage="hh_seed_preopt",
                                depth=0,
                                opt_method=str(adapt_inner_optimizer_key),
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
                        raise RuntimeError(
                            f"SciPy minimize is unavailable for {adapt_inner_optimizer_key} ADAPT inner optimizer."
                        )
                    seed_result = scipy_minimize(
                        _seed_obj,
                        theta_seed0,
                        method=str(adapt_inner_optimizer_key),
                        options=_scipy_inner_options(int(seed_maxiter)),
                    )
                    seed_theta = np.asarray(seed_result.x, dtype=float)
                    seed_energy = float(seed_result.fun)
                    seed_nfev = int(getattr(seed_result, "nfev", 0))
                    seed_nit = int(getattr(seed_result, "nit", 0))
                    seed_success = bool(getattr(seed_result, "success", False))
                    seed_message = str(getattr(seed_result, "message", ""))
                nfev_total += int(seed_nfev)

                selected_ops = list(seed_ops)
                selected_layout = _build_selected_layout(selected_ops)
                theta = np.asarray(seed_theta, dtype=float)
                if phase2_enabled:
                    if adapt_inner_optimizer_key == "SPSA":
                        phase2_optimizer_memory = phase2_memory_adapter.from_result(
                            seed_result,
                            method=str(adapt_inner_optimizer_key),
                            parameter_count=int(theta.size),
                            source="hh_seed_preopt",
                        )
                    else:
                        phase2_optimizer_memory = phase2_memory_adapter.unavailable(
                            method=str(adapt_inner_optimizer_key),
                            parameter_count=int(theta.size),
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

        phase1_prune_metadata_state, phase1_prune_first_seen_steps = _initialize_prune_metadata_state(
            labels_now=[str(op.label) for op in selected_ops],
            theta_logical_now=np.asarray(_logical_theta_alias(theta, _build_selected_layout(selected_ops)), dtype=float),
        )
        prune_summary = _default_prune_summary(reason="live_loop_init", energy=float(energy_current))

        rescue_cfg = RescueConfig(enabled=bool(phase3_enable_rescue_effective))

        def _phase3_try_rescue(
            *,
            psi_current_state: np.ndarray,
            shortlist_eval_records: list[dict[str, Any]],
            selected_position_append: int,
            history_rows: list[dict[str, Any]],
            trough_detected_now: bool,
        ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
            diagnostic = {
                "enabled": bool(phase3_enable_rescue_effective),
                "triggered": False,
                "reason": "disabled",
                "ranked": [],
                "selected_label": None,
                "selected_position": None,
                "overlap_gain": 0.0,
            }
            trigger_on, trigger_reason = should_trigger_rescue(
                enabled=bool(phase3_enable_rescue_effective),
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
                    psi_trial = _prepare_selected_state(
                        ops_now=ops_trial,
                        theta_now=theta_trial,
                        executor_now=(_build_compiled_executor(ops_trial) if adapt_state_backend_key == "compiled" else None),
                        parameter_layout_now=_build_selected_layout(ops_trial),
                    )
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

        beam_executor_memo: dict[tuple[str, ...], CompiledAnsatzExecutor] = {}
        beam_nfev_total = int(nfev_total)
        beam_branch_counter = 1
        beam_search_diagnostics: dict[str, Any] = {
            "beam_enabled": bool(beam_policy.beam_enabled),
            "live_branches": int(beam_policy.live_branches_effective),
            "children_per_parent": int(beam_policy.children_per_parent_effective),
            "terminated_keep": int(beam_policy.terminated_keep_effective),
            "fingerprint_version": "beam_scaffold_theta10_v1",
            "prune_key_version": "beam_energy_neg_score_burden_size_labels_theta10_id_v1",
            "admission_surface_version": "beam_phase3_shortlist_structural_stop_v1",
            "rounds": [],
        }

        def _compact_prune_audit(summary_raw: Mapping[str, Any] | None) -> dict[str, Any]:
            summary = dict(summary_raw) if isinstance(summary_raw, Mapping) else {}
            return {
                "enabled": bool(summary.get("enabled", False)),
                "permission_open": bool(summary.get("permission_open", False)),
                "permission_reason": str(summary.get("permission_reason", "unknown")),
                "executed": bool(summary.get("executed", False)),
                "accepted_count": int(summary.get("accepted_count", 0) or 0),
                "candidate_count": int(summary.get("candidate_count", 0) or 0),
                "selected_index": summary.get("selected_index"),
                "selected_label": summary.get("selected_label"),
                "snr_adm": float(summary.get("snr_adm", 0.0) or 0.0),
                "u_sat": float(summary.get("u_sat", 0.0) or 0.0),
                "probe_indices": [int(x) for x in summary.get("probe_indices", [])],
                "small_angle_pool_indices": [int(x) for x in summary.get("small_angle_pool_indices", [])],
            }

        def _surface_rows_summary(rows_raw: Sequence[Mapping[str, Any]] | None) -> dict[str, Any]:
            rows = [dict(row) for row in rows_raw if isinstance(row, Mapping)] if isinstance(rows_raw, Sequence) else []
            return {
                "count": int(len(rows)),
                "operator_labels": list(
                    dict.fromkeys(
                        str(row.get("candidate_label", ""))
                        for row in rows
                        if str(row.get("candidate_label", "")) != ""
                    )
                ),
                "generator_ids": list(
                    dict.fromkeys(
                        str(row.get("generator_id", ""))
                        for row in rows
                        if str(row.get("generator_id", "")) != ""
                    )
                ),
                "position_ids": list(
                    dict.fromkeys(
                        int(row.get("position_id"))
                        for row in rows
                        if row.get("position_id") is not None
                    )
                ),
                "runtime_split_modes": list(
                    dict.fromkeys(
                        str(row.get("runtime_split_mode", "off"))
                        for row in rows
                    )
                ),
            }

        def _selector_debug_row(record: Mapping[str, Any]) -> dict[str, Any]:
            rec = dict(record)
            feat_obj = rec.get("feature")

            def _feat_get(name: str, default: Any = None) -> Any:
                if isinstance(feat_obj, CandidateFeatures):
                    return getattr(feat_obj, name, default)
                if isinstance(feat_obj, Mapping):
                    return feat_obj.get(name, default)
                return default

            candidate_label_value = rec.get("candidate_label")
            if candidate_label_value in {None, ""}:
                candidate_label_value = _feat_get("candidate_label")
            if candidate_label_value in {None, ""}:
                candidate_label_value = _feat_get("label")

            return {
                "candidate_label": str(candidate_label_value or ""),
                "candidate_pool_index": int(rec.get("candidate_pool_index", -1)),
                "position_id": int(rec.get("position_id", -1)),
                "score_version": str(_feat_get("score_version", "")),
                "curvature_mode": str(_feat_get("curvature_mode", "")),
                "novelty_mode": str(_feat_get("novelty_mode", "")),
                "actual_fallback_mode": str(_feat_get("actual_fallback_mode", "")),
                "representation": str(_feat_get("runtime_split_chosen_representation", "parent")),
                "runtime_split_parent_label": _feat_get("runtime_split_parent_label"),
                "runtime_split_child_labels": [
                    str(x) for x in (_feat_get("runtime_split_child_labels", []) or [])
                ],
                "full_v2_score": float(rec.get("full_v2_score", float("-inf"))),
                "phase2_raw_score": float(rec.get("phase2_raw_score", float("-inf"))),
                "cheap_score": float(rec.get("cheap_score", rec.get("simple_score", float("-inf")))),
                "simple_score": float(rec.get("simple_score", float("-inf"))),
                "selector_score": (
                    float(_feat_get("selector_score"))
                    if _feat_get("selector_score") is not None
                    else None
                ),
                "g_abs": (
                    float(_feat_get("g_abs"))
                    if _feat_get("g_abs") is not None
                    else None
                ),
                "g_lcb": (
                    float(_feat_get("g_lcb"))
                    if _feat_get("g_lcb") is not None
                    else None
                ),
                "metric_proxy": (
                    float(_feat_get("metric_proxy"))
                    if _feat_get("metric_proxy") is not None
                    else None
                ),
                "F_raw": (
                    float(_feat_get("F_raw"))
                    if _feat_get("F_raw") is not None
                    else None
                ),
                "F_red": (
                    float(_feat_get("F_red"))
                    if _feat_get("F_red") is not None
                    else None
                ),
                "h_eff": (
                    float(_feat_get("h_eff"))
                    if _feat_get("h_eff") is not None
                    else None
                ),
                "ridge_used": (
                    float(_feat_get("ridge_used"))
                    if _feat_get("ridge_used") is not None
                    else None
                ),
                "selector_burden": (
                    float(_feat_get("selector_burden"))
                    if _feat_get("selector_burden") is not None
                    else None
                ),
                "phase2_raw_trust_gain": (
                    float(_feat_get("phase2_raw_trust_gain"))
                    if _feat_get("phase2_raw_trust_gain") is not None
                    else None
                ),
                "phase3_reduced_trust_gain": (
                    float(_feat_get("phase3_reduced_trust_gain"))
                    if _feat_get("phase3_reduced_trust_gain") is not None
                    else None
                ),
                "phase2_raw_novelty": (
                    float(_feat_get("phase2_raw_novelty"))
                    if _feat_get("phase2_raw_novelty") is not None
                    else None
                ),
                "phase3_reduced_novelty": (
                    float(_feat_get("phase3_reduced_novelty"))
                    if _feat_get("phase3_reduced_novelty") is not None
                    else None
                ),
                "phase2_burden_total": (
                    float(_feat_get("phase2_burden_total"))
                    if _feat_get("phase2_burden_total") is not None
                    else None
                ),
                "phase3_burden_total": (
                    float(_feat_get("phase3_burden_total"))
                    if _feat_get("phase3_burden_total") is not None
                    else None
                ),
                "compile_cost_total": (
                    float(_feat_get("compile_cost_total"))
                    if _feat_get("compile_cost_total") is not None
                    else None
                ),
                "depth_cost": (
                    float(_feat_get("depth_cost"))
                    if _feat_get("depth_cost") is not None
                    else None
                ),
                "new_group_cost": (
                    float(_feat_get("new_group_cost"))
                    if _feat_get("new_group_cost") is not None
                    else None
                ),
                "new_shot_cost": (
                    float(_feat_get("new_shot_cost"))
                    if _feat_get("new_shot_cost") is not None
                    else None
                ),
                "opt_dim_cost": (
                    float(_feat_get("opt_dim_cost"))
                    if _feat_get("opt_dim_cost") is not None
                    else None
                ),
                "reuse_count_cost": (
                    float(_feat_get("reuse_count_cost"))
                    if _feat_get("reuse_count_cost") is not None
                    else None
                ),
                "family_repeat_cost": (
                    float(_feat_get("family_repeat_cost"))
                    if _feat_get("family_repeat_cost") is not None
                    else None
                ),
                "motif_bonus": (
                    float(_feat_get("motif_bonus"))
                    if _feat_get("motif_bonus") is not None
                    else None
                ),
                "compatibility_penalty_total": (
                    float(_feat_get("compatibility_penalty_total"))
                    if _feat_get("compatibility_penalty_total") is not None
                    else None
                ),
            }

        def _selector_debug_rows(
            rows_raw: Sequence[Mapping[str, Any]] | None,
            *,
            topk: int,
        ) -> list[dict[str, Any]]:
            rows = [dict(row) for row in rows_raw if isinstance(row, Mapping)] if isinstance(rows_raw, Sequence) else []
            if int(topk) <= 0 or not rows:
                return []
            rows_sorted = sorted(rows, key=_phase2_record_sort_key)[: int(topk)]
            return [_selector_debug_row(row) for row in rows_sorted]

        def _selector_debug_enabled_for_depth(depth_one_based: int) -> bool:
            if int(phase3_selector_debug_topk_val) <= 0:
                return False
            if int(phase3_selector_debug_max_depth_val) <= 0:
                return True
            return int(depth_one_based) <= int(phase3_selector_debug_max_depth_val)

        def _selector_debug_payload(
            *,
            depth_one_based: int,
            beam_enabled: bool,
            selection_mode_value: str,
            stage_name_value: str,
            selected_feature_row: Mapping[str, Any] | None,
            scored_rows: Sequence[Mapping[str, Any]] | None,
            phase2_rows: Sequence[Mapping[str, Any]] | None,
            phase3_rows: Sequence[Mapping[str, Any]] | None,
            admitted_rows: Sequence[Mapping[str, Any]] | None,
            split_summary: Mapping[str, Any] | None,
        ) -> dict[str, Any]:
            return {
                "depth": int(depth_one_based),
                "beam_enabled": bool(beam_enabled),
                "selection_mode": str(selection_mode_value),
                "stage_name": str(stage_name_value),
                "score_config": {
                    "lambda_H": float(getattr(phase2_score_cfg, "lambda_H", 0.0)),
                    "rho": float(getattr(phase2_score_cfg, "rho", 0.0)),
                    "gamma_N": float(getattr(phase2_score_cfg, "gamma_N", 0.0)),
                    "phase2_frontier_ratio": float(getattr(phase2_score_cfg, "phase2_frontier_ratio", 0.0)),
                    "phase3_frontier_ratio": float(getattr(phase2_score_cfg, "phase3_frontier_ratio", 0.0)),
                    "batching_enabled": bool(phase2_enable_batching),
                    "batch_target_size": int(getattr(phase2_score_cfg, "batch_target_size", 0)),
                    "batch_size_cap": int(getattr(phase2_score_cfg, "batch_size_cap", 0)),
                    "batch_near_degenerate_ratio": float(
                        getattr(phase2_score_cfg, "batch_near_degenerate_ratio", 0.0)
                    ),
                    "batch_rank_rel_tol": float(getattr(phase2_score_cfg, "batch_rank_rel_tol", 0.0)),
                    "batch_additivity_tol": float(getattr(phase2_score_cfg, "batch_additivity_tol", 0.0)),
                    "w_depth": float(getattr(phase2_score_cfg, "wD", 0.0)),
                    "w_group": float(getattr(phase2_score_cfg, "wG", 0.0)),
                    "w_shot": float(getattr(phase2_score_cfg, "wC", 0.0)),
                    "w_optdim": float(getattr(phase2_score_cfg, "wP", 0.0)),
                    "w_reuse": float(getattr(phase2_score_cfg, "wc", 0.0)),
                    "w_lifetime": float(getattr(phase2_score_cfg, "lifetime_weight", 0.0)),
                    "lambda_F": float(getattr(phase2_score_cfg, "lambda_F", 0.0)),
                    "score_z_alpha": float(getattr(phase2_score_cfg, "z_alpha", 0.0)),
                    "eta_L": float(getattr(phase2_score_cfg, "eta_L", 0.0)),
                    "depth_ref": float(getattr(phase2_score_cfg, "depth_ref", 1.0)),
                    "group_ref": float(getattr(phase2_score_cfg, "group_ref", 1.0)),
                    "shot_ref": float(getattr(phase2_score_cfg, "shot_ref", 1.0)),
                    "optdim_ref": float(getattr(phase2_score_cfg, "optdim_ref", 1.0)),
                    "reuse_ref": float(getattr(phase2_score_cfg, "reuse_ref", 1.0)),
                    "family_ref": float(getattr(phase2_score_cfg, "family_ref", 1.0)),
                    "novelty_eps": float(getattr(phase2_score_cfg, "novelty_eps", 0.0)),
                    "cheap_score_eps": float(getattr(phase2_score_cfg, "cheap_score_eps", 0.0)),
                    "metric_floor": float(getattr(phase2_score_cfg, "metric_floor", 0.0)),
                    "reduced_metric_collapse_rel_tol": float(
                        getattr(phase2_score_cfg, "reduced_metric_collapse_rel_tol", 0.0)
                    ),
                    "ridge_growth_factor": float(getattr(phase2_score_cfg, "ridge_growth_factor", 0.0)),
                    "ridge_max_steps": int(getattr(phase2_score_cfg, "ridge_max_steps", 0)),
                    "leakage_cap": float(getattr(phase2_score_cfg, "leakage_cap", 0.0)),
                    "motif_bonus_weight": float(getattr(phase2_score_cfg, "motif_bonus_weight", 0.0)),
                    "duplicate_penalty_weight": float(
                        getattr(phase2_score_cfg, "duplicate_penalty_weight", 0.0)
                    ),
                    "compat_overlap_weight": float(
                        getattr(phase2_score_cfg, "compat_overlap_weight", 0.0)
                    ),
                    "compat_comm_weight": float(getattr(phase2_score_cfg, "compat_comm_weight", 0.0)),
                    "compat_curv_weight": float(getattr(phase2_score_cfg, "compat_curv_weight", 0.0)),
                    "compat_sched_weight": float(getattr(phase2_score_cfg, "compat_sched_weight", 0.0)),
                    "compat_measure_weight": float(
                        getattr(phase2_score_cfg, "compat_measure_weight", 0.0)
                    ),
                    "remaining_evaluations_proxy_mode": str(phase2_remaining_evaluations_proxy_mode),
                    "lifetime_cost_mode": str(phase3_lifetime_cost_mode_key),
                    "runtime_split_mode": str(phase3_runtime_split_mode_key),
                },
                "runtime_split_summary": (
                    dict(split_summary) if isinstance(split_summary, Mapping) else {}
                ),
                "selected": (
                    _selector_debug_row({"feature": dict(selected_feature_row), **dict(selected_feature_row)})
                    if isinstance(selected_feature_row, Mapping)
                    else None
                ),
                "scored_topk": _selector_debug_rows(
                    scored_rows,
                    topk=int(phase3_selector_debug_topk_val),
                ),
                "phase2_shortlist_topk": _selector_debug_rows(
                    phase2_rows,
                    topk=int(phase3_selector_debug_topk_val),
                ),
                "phase3_shortlist_topk": _selector_debug_rows(
                    phase3_rows,
                    topk=int(phase3_selector_debug_topk_val),
                ),
                "admitted_topk": _selector_debug_rows(
                    admitted_rows,
                    topk=int(phase3_selector_debug_topk_val),
                ),
            }

        def _phase3_surface_audit_payload(
            *,
            scored_rows: Sequence[Mapping[str, Any]] | None,
            retained_rows: Sequence[Mapping[str, Any]] | None,
            admitted_rows: Sequence[Mapping[str, Any]] | None,
            beam_enabled: bool,
        ) -> dict[str, Any]:
            return {
                "scored_surface_notation": ("R_3(b)" if beam_enabled else "R_3(t)"),
                "scored_surface_key": "phase2_scored_rows",
                "scored_surface_semantics": "last_scored_candidate_surface",
                "retained_shortlist_notation": ("S_3(b)" if beam_enabled else "S_3(t)"),
                "retained_shortlist_key": "phase2_retained_shortlist_rows",
                "retained_shortlist_semantics": "controller_retained_shortlist",
                "admitted_set_notation": ("A_b" if beam_enabled else "B_t^*"),
                "admitted_set_key": "phase2_admitted_rows",
                "admitted_set_semantics": (
                    "branch_local_retained_admission_set"
                    if beam_enabled
                    else "reduced_plane_admitted_set"
                ),
                "scored_surface": _surface_rows_summary(scored_rows),
                "retained_shortlist": _surface_rows_summary(retained_rows),
                "admitted_set": _surface_rows_summary(admitted_rows),
            }

        def _active_hh_pool_summary_payload(
            *,
            phase1_rows: Sequence[Mapping[str, Any]] | None,
            phase2_rows: Sequence[Mapping[str, Any]] | None,
            phase3_rows: Sequence[Mapping[str, Any]] | None,
        ) -> dict[str, Any]:
            phase1_rows_list = [dict(row) for row in phase1_rows if isinstance(row, Mapping)] if isinstance(phase1_rows, Sequence) else []
            phase2_rows_list = [dict(row) for row in phase2_rows if isinstance(row, Mapping)] if isinstance(phase2_rows, Sequence) else []
            phase3_rows_list = [dict(row) for row in phase3_rows if isinstance(row, Mapping)] if isinstance(phase3_rows, Sequence) else []
            phase1_summary = _surface_rows_summary(phase1_rows_list)
            phase2_summary = _surface_rows_summary(phase2_rows_list)
            phase3_summary = _surface_rows_summary(phase3_rows_list)

            def _split_closed_labels(
                seed_labels: set[str],
                rows_extra: Sequence[Mapping[str, Any]],
            ) -> set[str]:
                closed = set(str(x) for x in seed_labels)
                changed = True
                while changed:
                    changed = False
                    for row in rows_extra:
                        label = str(row.get("candidate_label", ""))
                        parent_label = str(row.get("runtime_split_parent_label", ""))
                        if not label:
                            continue
                        if parent_label and parent_label in closed and label not in closed:
                            closed.add(label)
                            changed = True
                return closed

            phase1_labels_raw = set(str(x) for x in phase1_summary.get("operator_labels", []))
            phase2_labels_raw = set(str(x) for x in phase2_summary.get("operator_labels", []))
            phase3_labels_raw = set(str(x) for x in phase3_summary.get("operator_labels", []))
            phase1_labels_effective = _split_closed_labels(
                phase1_labels_raw,
                [*phase2_rows_list, *phase3_rows_list],
            )
            phase2_labels_effective = _split_closed_labels(phase2_labels_raw, phase3_rows_list)
            phase3_labels_effective = set(phase3_labels_raw)
            phase1_summary["generator_image_labels_effective"] = sorted(phase1_labels_effective)
            phase1_summary["generator_image_count_effective"] = int(len(phase1_labels_effective))
            phase2_summary["generator_image_labels_effective"] = sorted(phase2_labels_effective)
            phase2_summary["generator_image_count_effective"] = int(len(phase2_labels_effective))
            phase3_summary["generator_image_labels_effective"] = sorted(phase3_labels_effective)
            phase3_summary["generator_image_count_effective"] = int(len(phase3_labels_effective))
            return {
                "summary_label": "Omega_HH_active",
                "omega_chain": ["Omega_HH^(1)", "Omega_HH^(2)", "Omega_HH^(3)"],
                "nested_generator_image_inclusion": {
                    "phase2_in_phase1": bool(phase2_labels_effective.issubset(phase1_labels_effective)),
                    "phase3_in_phase2": bool(phase3_labels_effective.issubset(phase2_labels_effective)),
                    "phase3_in_phase1": bool(phase3_labels_effective.issubset(phase1_labels_effective)),
                },
                "phases": {
                    "phase1": {
                        "omega_label": "Omega_HH^(1)",
                        "generator_image_notation": "G^(1)",
                        "rows_key": "phase1_retained_rows",
                        "rows_semantics": "phase1_retained_record_shortlist",
                        "generator_family_notation": "G_adapt^(1)",
                        **dict(phase1_summary),
                    },
                    "phase2": {
                        "omega_label": "Omega_HH^(2)",
                        "generator_image_notation": "G^(2)",
                        "rows_key": "phase2_geometric_shortlist_rows",
                        "rows_semantics": "phase2_retained_geometric_shortlist",
                        "generator_family_notation": "G_adapt^(2)",
                        **dict(phase2_summary),
                    },
                    "phase3": {
                        "omega_label": "Omega_HH^(3)",
                        "generator_image_notation": "G^(3)",
                        "rows_key": "phase2_retained_shortlist_rows",
                        "rows_semantics": "phase3_retained_shortlist_generator_image",
                        "generator_family_notation": "G_adapt^(3)",
                        **dict(phase3_summary),
                    },
                },
            }

        def _controller_snapshot_payload(snapshot_raw: Any | None) -> dict[str, Any] | None:
            if not isinstance(snapshot_raw, PhaseControllerSnapshot):
                return None
            return {
                "snapshot_version": str(snapshot_raw.snapshot_version),
                "step_index": int(snapshot_raw.step_index),
                "depth_local": int(snapshot_raw.depth_local),
                "depth_left": int(snapshot_raw.depth_left),
                "runway_ratio": float(snapshot_raw.runway_ratio),
                "early_coordinate": float(snapshot_raw.early_coordinate),
                "late_coordinate": float(snapshot_raw.late_coordinate),
                "frontier_ratio": float(snapshot_raw.frontier_ratio),
                "phase_thresholds": {
                    str(k): float(v) for k, v in dict(snapshot_raw.phase_thresholds).items()
                },
                "phase_caps": {
                    str(k): int(v) for k, v in dict(snapshot_raw.phase_caps).items()
                },
                "phase_shots": {
                    str(k): int(v) for k, v in dict(snapshot_raw.phase_shots).items()
                },
                "phase_uncertainty": {
                    str(k): float(v) for k, v in dict(snapshot_raw.phase_uncertainty).items()
                },
            }

        def _controller_telemetry_summary_payload(
            *,
            stage_name: str | None,
            residual_opened: bool,
            last_probe_reason: str | None,
            stage_events: Sequence[Mapping[str, Any]] | None,
            last_snapshot: Any | None,
        ) -> dict[str, Any]:
            stage_rows = (
                [dict(row) for row in stage_events if isinstance(row, Mapping)]
                if isinstance(stage_events, Sequence)
                else []
            )
            return {
                "telemetry_label": "T_b^ctrl",
                "stage_name": (None if stage_name is None else str(stage_name)),
                "residual_opened": bool(residual_opened),
                "last_probe_reason": (None if last_probe_reason is None else str(last_probe_reason)),
                "stage_event_count": int(len(stage_rows)),
                "last_stage_event": (dict(stage_rows[-1]) if stage_rows else None),
                "last_snapshot": _controller_snapshot_payload(last_snapshot),
            }

        def _branch_state_summary_payload(
            *,
            beam_enabled: bool,
            branch_id: int | None,
            parent_branch_id: int | None,
            history_rows: Sequence[Mapping[str, Any]] | None,
            depth_local: int,
            ansatz_depth: int,
            terminated: bool,
            termination_label: str | None,
            cumulative_selector_score: float,
            cumulative_selector_burden: float,
            stage_name: str | None,
            residual_opened: bool,
            last_probe_reason: str | None,
            stage_events: Sequence[Mapping[str, Any]] | None,
            last_snapshot: Any | None,
        ) -> dict[str, Any]:
            rows = (
                [dict(row) for row in history_rows if isinstance(row, Mapping)]
                if isinstance(history_rows, Sequence)
                else []
            )
            return {
                "branch_state_notation": "\\mathfrak b_*",
                "status": ("terminal" if bool(terminated) else "frontier"),
                "termination_label": (
                    str(termination_label) if bool(terminated) and termination_label is not None else None
                ),
                "beam_enabled": bool(beam_enabled),
                "branch_id": (None if branch_id is None else int(branch_id)),
                "parent_branch_id": (
                    None if parent_branch_id is None else int(parent_branch_id)
                ),
                "depth_local": int(depth_local),
                "history_step_count": int(len(rows)),
                "ansatz_depth": int(ansatz_depth),
                "cumulative_selector_score": float(cumulative_selector_score),
                "cumulative_selector_burden": float(cumulative_selector_burden),
                "controller_telemetry": _controller_telemetry_summary_payload(
                    stage_name=stage_name,
                    residual_opened=bool(residual_opened),
                    last_probe_reason=last_probe_reason,
                    stage_events=stage_events,
                    last_snapshot=last_snapshot,
                ),
            }

        def _scaffold_fingerprint_payload(
            *,
            operator_labels: Sequence[str],
            generator_ids: Sequence[str],
            num_parameters: int,
        ) -> dict[str, Any]:
            payload = {
                "selected_operator_labels": [str(x) for x in operator_labels],
                "selected_generator_ids": [str(x) for x in generator_ids if str(x) != ""],
                "num_parameters": int(num_parameters),
            }
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
            return {
                "fingerprint_notation": "fp(O_*)",
                "fingerprint_version": "scaffold_labels_generator_ids_params_v1",
                "fingerprint_sha256": str(digest),
                **payload,
            }

        def _optimizer_memory_contract_summary_payload(
            *,
            beam_enabled: bool,
            branch_id: int | None,
            memory_state: Mapping[str, Any] | None,
            operator_labels: Sequence[str],
            generator_ids: Sequence[str],
            num_parameters: int,
            last_active_subset_source: str | None,
            last_active_subset_reused: bool,
        ) -> dict[str, Any]:
            state = dict(memory_state) if isinstance(memory_state, Mapping) else {}
            remap_events = [
                dict(row)
                for row in state.get("remap_events", [])
                if isinstance(row, Mapping)
            ]
            structural_transport = any(
                str(row.get("op", "")) in {"insert", "remove"} for row in remap_events
            )
            memory_source_value = (
                str(state.get("source"))
                if state.get("source") not in {None, ""}
                else (
                    str(last_active_subset_source)
                    if last_active_subset_source not in {None, ""}
                    else "unavailable"
                )
            )
            if not bool(state.get("available", False)):
                observed_transport_mode = "unavailable"
            elif structural_transport:
                observed_transport_mode = "canonical_embedding_or_index_remap"
            else:
                observed_transport_mode = "same_scaffold_active_subset"
            return {
                "contract_label": "phase2_optimizer_memory_contract",
                "beam_enabled": bool(beam_enabled),
                "branch_id": (None if branch_id is None else int(branch_id)),
                "scaffold_fingerprint": _scaffold_fingerprint_payload(
                    operator_labels=operator_labels,
                    generator_ids=generator_ids,
                    num_parameters=int(num_parameters),
                ),
                "exact_reuse_rule": "requires_matching_scaffold_fingerprint",
                "fingerprint_match_required": True,
                "canonical_embedding_notation": "theta -> theta⊕_p 0",
                "refit_window_notation": "W(r;t)",
                "memory_available": bool(state.get("available", False)),
                "memory_optimizer": str(state.get("optimizer", "unknown")),
                "memory_parameter_count": int(state.get("parameter_count", max(0, int(num_parameters)))),
                "memory_source": str(memory_source_value),
                "last_active_subset_source": (
                    None
                    if last_active_subset_source in {None, ""}
                    else str(last_active_subset_source)
                ),
                "last_active_subset_reused": bool(last_active_subset_reused),
                "structural_transport_detected": bool(structural_transport),
                "observed_transport_mode": str(observed_transport_mode),
                "remap_event_count": int(len(remap_events)),
                "remap_event_tail": [dict(row) for row in remap_events[-8:]],
            }

        def _controller_runtime_boundary_summary_payload(
            *,
            phase_enabled: bool,
            cfg: StageControllerConfig,
            stage_controller_payload: Mapping[str, Any] | None,
            current_snapshot_payload: Mapping[str, Any] | None,
            beam_enabled: bool,
            branch_id: int | None,
        ) -> dict[str, Any]:
            return {
                "summary_label": "appendix_a_runtime_boundary",
                "beam_enabled": bool(beam_enabled),
                "branch_id": (None if branch_id is None else int(branch_id)),
                "phase_enabled": bool(phase_enabled),
                "symbolic_result_keys": [
                    "selected_scaffold_summary",
                    "selected_scaffold_final_choice",
                    "selected_scaffold_branch_state",
                    "selected_state_summary",
                    "selected_scaffold_history",
                    "selected_scaffold_record_chain",
                    "active_hh_pool_summary",
                    "active_phase3_surface_summary",
                ],
                "runtime_controller_keys": [
                    "stage_controller",
                    "selected_scaffold_branch_state.controller_telemetry",
                    "selected_scaffold_optimizer_memory_contract",
                ],
                "runtime_law_notation": {
                    "thresholds": "tau_k(t)",
                    "caps": "N_k(t)",
                    "shots_phase1": "N_shot,1(t)",
                    "shots_phasek": "N_shot,k(t)",
                },
                "runtime_dependencies": [
                    "available_depth",
                    "wall_clock",
                    "sampling_budget",
                    "device_noise",
                ],
                "calibration_status": "runtime_calibrated_not_symbolic",
                "configured_bounds": {
                    "tau_phase1_min": float(cfg.tau_phase1_min),
                    "tau_phase1_max": float(cfg.tau_phase1_max),
                    "tau_phase2_min": float(cfg.tau_phase2_min),
                    "tau_phase2_max": float(cfg.tau_phase2_max),
                    "tau_phase3_min": float(cfg.tau_phase3_min),
                    "tau_phase3_max": float(cfg.tau_phase3_max),
                    "cap_phase1_min": int(cfg.cap_phase1_min),
                    "cap_phase1_max": int(cfg.cap_phase1_max),
                    "cap_phase2_min": int(cfg.cap_phase2_min),
                    "cap_phase2_max": int(cfg.cap_phase2_max),
                    "cap_phase3_min": int(cfg.cap_phase3_min),
                    "cap_phase3_max": int(cfg.cap_phase3_max),
                    "shot_min": int(cfg.shot_min),
                    "shot_max": int(cfg.shot_max),
                },
                "stage_controller_payload": (
                    dict(stage_controller_payload)
                    if isinstance(stage_controller_payload, Mapping)
                    else None
                ),
                "current_controller_snapshot": (
                    dict(current_snapshot_payload)
                    if isinstance(current_snapshot_payload, Mapping)
                    else None
                ),
            }

        def _beam_branch_summary(branch: _BeamBranchState) -> dict[str, Any]:
            prune_history = [
                _compact_prune_audit(row.get("post_admission_prune"))
                for row in branch.history
                if isinstance(row, Mapping) and isinstance(row.get("post_admission_prune"), Mapping)
            ]
            branch_controller_snapshot = branch.phase1_stage.snapshot().get("last_snapshot")
            branch_generator_ids = [
                str(meta.get("generator_id", ""))
                for meta in selected_generator_metadata_for_labels(
                    [str(op.label) for op in branch.selected_ops],
                    pool_generator_registry,
                )
                if str(meta.get("generator_id", "")) != ""
            ]
            return {
                "branch_id": int(branch.branch_id),
                "parent_branch_id": (
                    None if branch.parent_branch_id is None else int(branch.parent_branch_id)
                ),
                "depth_local": int(branch.depth_local),
                "stop_reason": (None if branch.stop_reason is None else str(branch.stop_reason)),
                "terminated": bool(branch.terminated),
                "status": ("terminal" if bool(branch.terminated) else "frontier"),
                "termination_label": (
                    str(branch.stop_reason)
                    if bool(branch.terminated) and branch.stop_reason is not None
                    else None
                ),
                "last_transition_kind": str(branch.last_transition_kind),
                "last_admission_record_count": int(branch.last_admission_record_count),
                "energy": float(branch.energy_current),
                "cumulative_selector_score": float(branch.cumulative_selector_score),
                "cumulative_selector_burden": float(branch.cumulative_selector_burden),
                "scored_surface_count": int(len(branch.phase2_last_shortlist_records)),
                "retained_shortlist_count": int(len(branch.phase2_last_retained_shortlist_records)),
                "admitted_count": int(len(branch.phase2_last_admitted_records)),
                "phase3_surface_summary": _phase3_surface_audit_payload(
                    scored_rows=branch.phase2_last_shortlist_records,
                    retained_rows=branch.phase2_last_retained_shortlist_records,
                    admitted_rows=branch.phase2_last_admitted_records,
                    beam_enabled=True,
                ),
                "prune_key": dict(_beam_prune_key_payload(branch)),
                "last_prune": _compact_prune_audit(branch.phase1_last_prune_summary),
                "prune_history": [dict(x) for x in prune_history],
                "branch_state_summary": _branch_state_summary_payload(
                    beam_enabled=True,
                    branch_id=int(branch.branch_id),
                    parent_branch_id=(
                        None if branch.parent_branch_id is None else int(branch.parent_branch_id)
                    ),
                    history_rows=branch.history,
                    depth_local=int(branch.depth_local),
                    ansatz_depth=int(len(branch.selected_ops)),
                    terminated=bool(branch.terminated),
                    termination_label=(
                        None if branch.stop_reason is None else str(branch.stop_reason)
                    ),
                    cumulative_selector_score=float(branch.cumulative_selector_score),
                    cumulative_selector_burden=float(branch.cumulative_selector_burden),
                    stage_name=str(branch.phase1_stage.stage_name),
                    residual_opened=bool(branch.phase1_residual_opened),
                    last_probe_reason=str(branch.phase1_last_probe_reason),
                    stage_events=branch.phase1_stage_events,
                    last_snapshot=branch_controller_snapshot,
                ),
                "optimizer_memory_contract_summary": _optimizer_memory_contract_summary_payload(
                    beam_enabled=True,
                    branch_id=int(branch.branch_id),
                    memory_state=branch.phase2_optimizer_memory,
                    operator_labels=[str(op.label) for op in branch.selected_ops],
                    generator_ids=branch_generator_ids,
                    num_parameters=int(np.asarray(branch.theta, dtype=float).size),
                    last_active_subset_source=str(branch.phase2_last_optimizer_memory_source),
                    last_active_subset_reused=bool(branch.phase2_last_optimizer_memory_reused),
                ),
            }

        def _beam_clone_branch(
            branch: _BeamBranchState,
            *,
            branch_id: int,
            parent_branch_id: int | None,
        ) -> _BeamBranchState:
            cloned = branch.clone_for_child(branch_id=int(branch_id))
            cloned.parent_branch_id = (
                None if parent_branch_id is None else int(parent_branch_id)
            )
            return cloned

        def _beam_executor_key(ops: Sequence[AnsatzTerm]) -> tuple[str, ...]:
            return tuple(str(op.label) for op in ops)

        def _get_beam_executor(ops: Sequence[AnsatzTerm]) -> CompiledAnsatzExecutor | None:
            if adapt_state_backend_key != "compiled" or len(ops) == 0:
                return None
            key = _beam_executor_key(ops)
            executor = beam_executor_memo.get(key)
            if executor is None:
                executor = _build_compiled_executor(list(ops))
                beam_executor_memo[key] = executor
            return executor

        def _beam_label_signature(ops: Sequence[AnsatzTerm]) -> tuple[str, ...]:
            return tuple(str(op.label) for op in ops)

        def _beam_round10_theta(theta_now: np.ndarray) -> tuple[float, ...]:
            theta_vec = np.asarray(theta_now, dtype=float).reshape(-1)
            return tuple(round(float(x), 10) for x in theta_vec.tolist())

        def _branch_state_fingerprint(branch: _BeamBranchState) -> str:
            payload = {
                "depth_local": int(branch.depth_local),
                "labels": list(_beam_label_signature(branch.selected_ops)),
                "theta_round10": list(_beam_round10_theta(branch.theta)),
            }
            return hashlib.sha256(
                json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()

        def _proposal_fingerprint(
            *,
            parent: _BeamBranchState,
            plan: _BranchExpansionPlan,
        ) -> str:
            payload = {
                "parent": _branch_state_fingerprint(parent),
                "candidate_pool_index": int(plan.candidate_pool_index),
                "position_id": int(plan.position_id),
                "selection_mode": str(plan.selection_mode),
                "candidate_label": str(plan.candidate_label),
                "init_theta": round(float(plan.init_theta), 12),
            }
            return hashlib.sha256(
                json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()

        def _branch_optimizer_seed(
            *,
            base_seed: int,
            stage_tag: str,
            depth_local: int,
            parent_state_fingerprint: str,
            proposal_fingerprint: str | None,
        ) -> int:
            payload = {
                "base_seed": int(base_seed),
                "stage_tag": str(stage_tag),
                "depth_local": int(depth_local),
                "parent_state_fingerprint": str(parent_state_fingerprint),
                "proposal_fingerprint": (None if proposal_fingerprint is None else str(proposal_fingerprint)),
            }
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode("utf-8")
            ).digest()
            return int.from_bytes(digest[:8], "big") % (2**31 - 1)

        def _beam_prune_key_payload(branch: _BeamBranchState) -> dict[str, Any]:
            return {
                "energy": float(branch.energy_current),
                "cumulative_selector_score": float(branch.cumulative_selector_score),
                "cumulative_selector_burden": float(branch.cumulative_selector_burden),
                "ansatz_depth": int(len(branch.selected_ops)),
                "labels": list(_beam_label_signature(branch.selected_ops)),
                "theta_round10": [float(x) for x in _beam_round10_theta(branch.theta)],
                "theta_round10_digits": 10,
                "branch_id": int(branch.branch_id),
            }

        def _beam_prune_key(branch: _BeamBranchState) -> tuple[Any, ...]:
            payload = _beam_prune_key_payload(branch)
            return (
                float(payload["energy"]),
                -float(payload["cumulative_selector_score"]),
                float(payload["cumulative_selector_burden"]),
                int(payload["ansatz_depth"]),
                tuple(str(x) for x in payload["labels"]),
                tuple(float(x) for x in payload["theta_round10"]),
                int(payload["branch_id"]),
            )

        def _beam_dedup(branches: Sequence[_BeamBranchState]) -> list[_BeamBranchState]:
            keep: dict[str, _BeamBranchState] = {}
            for branch in branches:
                fingerprint = _branch_state_fingerprint(branch)
                incumbent = keep.get(fingerprint)
                if incumbent is None or _beam_prune_key(branch) < _beam_prune_key(incumbent):
                    keep[fingerprint] = branch
            return list(keep.values())

        def _beam_prune(
            branches: Sequence[_BeamBranchState],
            *,
            cap: int,
        ) -> list[_BeamBranchState]:
            deduped = _beam_dedup(branches)
            return sorted(deduped, key=_beam_prune_key)[: int(max(0, cap))]

        def _evaluate_beam_branch(
            branch: _BeamBranchState,
            *,
            depth: int,
            children_cap: int,
        ) -> _BranchStepScratch:
            nonlocal beam_nfev_total

            executor = _get_beam_executor(branch.selected_ops)
            psi_current_local = _prepare_selected_state(
                ops_now=list(branch.selected_ops),
                theta_now=np.asarray(branch.theta, dtype=float),
                executor_now=executor,
                parameter_layout_now=_build_selected_layout(list(branch.selected_ops)),
            )
            energy_current_local, hpsi_current_local = energy_via_one_apply(
                psi_current_local,
                h_compiled,
            )
            energy_current_exact_local = float(energy_current_local)
            energy_current_local = (
                float(branch.energy_current)
                if phase3_oracle_inner_objective_enabled
                else float(energy_current_exact_local)
            )
            branch_layout = _build_selected_layout(list(branch.selected_ops))
            theta_logical_branch = _logical_theta_alias(
                np.asarray(branch.theta, dtype=float),
                branch_layout,
            )

            gradient_eval_t0 = time.perf_counter()
            gradients_local = np.zeros(len(pool), dtype=float)
            grad_magnitudes_local = np.zeros(len(pool), dtype=float)
            available_indices_local = set(int(x) for x in branch.available_indices)
            for idx in available_indices_local:
                apsi = _apply_compiled_polynomial(psi_current_local, pool_compiled[int(idx)])
                gradients_local[int(idx)] = adapt_commutator_grad_from_hpsi(
                    hpsi_current_local,
                    apsi,
                )
                grad_magnitudes_local[int(idx)] = abs(float(gradients_local[int(idx)]))
            gradient_eval_elapsed_s_local = float(time.perf_counter() - gradient_eval_t0)

            if available_indices_local:
                max_grad_local = float(
                    max(float(grad_magnitudes_local[int(idx)]) for idx in available_indices_local)
                )
            else:
                max_grad_local = 0.0

            append_position_local = int(branch_layout.logical_parameter_count)
            selected_position_local = int(append_position_local)
            best_idx_local = int(sorted(available_indices_local)[0]) if available_indices_local else -1
            stage_name_local = str(branch.phase1_stage.stage_name)
            phase1_feature_selected_local: dict[str, Any] | None = None
            phase1_stage_transition_reason_local = "legacy"
            phase1_last_probe_reason_local = "append_only"
            phase1_last_positions_considered_local = [int(append_position_local)]
            phase1_last_trough_detected_local = False
            phase1_last_trough_probe_triggered_local = False
            phase1_last_selected_score_local: float | None = None
            phase1_last_retained_records_local: list[dict[str, Any]] = []
            phase2_last_shortlist_records_local: list[dict[str, Any]] = []
            phase2_last_geometric_shortlist_records_local: list[dict[str, Any]] = []
            phase2_last_retained_shortlist_records_local: list[dict[str, Any]] = []
            phase2_last_admitted_records_local: list[dict[str, Any]] = []
            phase2_last_batch_selected_local = False
            phase2_last_batch_penalty_total_local = 0.0
            phase2_last_optimizer_memory_reused_local = False
            phase2_last_optimizer_memory_source_local = "unavailable"
            phase2_last_shortlist_eval_records_local: list[dict[str, Any]] = []
            phase3_runtime_split_summary_local = copy.deepcopy(
                branch.phase3_runtime_split_summary
            )
            phase1_stage_local = branch.phase1_stage.clone()
            phase1_stage_events_local = [dict(x) for x in branch.phase1_stage_events]
            phase1_residual_opened_local = bool(branch.phase1_residual_opened)
            selection_mode_local = "simple_v1"
            candidate_plans: list[_BranchExpansionPlan] = []
            fallback_scan_size_local = 0
            fallback_best_probe_delta_e_local: float | None = None
            fallback_best_probe_theta_local: float | None = None

            if not available_indices_local and not allow_repeats:
                return _BranchStepScratch(
                    energy_current=energy_current_local,
                    psi_current=np.asarray(psi_current_local, dtype=complex),
                    hpsi_current=np.asarray(hpsi_current_local, dtype=complex),
                    gradients=np.asarray(gradients_local, dtype=float),
                    grad_magnitudes=np.asarray(grad_magnitudes_local, dtype=float),
                    max_grad=float(max_grad_local),
                    gradient_eval_elapsed_s=float(gradient_eval_elapsed_s_local),
                    append_position=int(append_position_local),
                    best_idx=int(best_idx_local),
                    selected_position=int(selected_position_local),
                    selection_mode="pool_exhausted",
                    stage_name=str(stage_name_local),
                    phase1_feature_selected=None,
                    phase1_stage_transition_reason="pool_exhausted",
                    phase1_stage_now=str(phase1_stage_local.stage_name),
                    phase1_stage_after_transition=phase1_stage_local,
                    phase1_last_probe_reason=str(phase1_last_probe_reason_local),
                    phase1_last_positions_considered=list(phase1_last_positions_considered_local),
                    phase1_last_trough_detected=bool(phase1_last_trough_detected_local),
                    phase1_last_trough_probe_triggered=bool(phase1_last_trough_probe_triggered_local),
                    phase1_last_selected_score=phase1_last_selected_score_local,
                    phase1_last_retained_records=[],
                    phase2_last_shortlist_records=[],
                    phase2_last_geometric_shortlist_records=[],
                    phase2_last_retained_shortlist_records=[],
                    phase2_last_admitted_records=[],
                    phase2_last_batch_selected=False,
                    phase2_last_batch_penalty_total=0.0,
                    phase2_last_optimizer_memory_reused=False,
                    phase2_last_optimizer_memory_source="unavailable",
                    phase2_last_shortlist_eval_records=[],
                    phase1_residual_opened=bool(phase1_residual_opened_local),
                    available_indices_after_transition=set(available_indices_local),
                    phase1_stage_events_after_transition=[dict(x) for x in phase1_stage_events_local],
                    phase3_runtime_split_summary_after_eval=copy.deepcopy(phase3_runtime_split_summary_local),
                    proposals=[],
                    stop_reason="pool_exhausted",
                    fallback_scan_size=0,
                    fallback_best_probe_delta_e=None,
                    fallback_best_probe_theta=None,
                )

            available_sorted = sorted(
                list(available_indices_local),
                key=lambda idx: -float(grad_magnitudes_local[int(idx)]),
            )
            shortlist_local = available_sorted[: min(len(available_sorted), 64)]
            candidate_metric_cache_local: dict[int, float] = {}
            family_repeat_cache_local: dict[str, float] = {}

            def _evaluate_phase1_positions_local(
                positions_considered_local: list[int],
                *,
                trough_probe_triggered_local: bool,
            ) -> dict[str, Any]:
                best_score_local = float("-inf")
                best_idx_inner = int(shortlist_local[0]) if shortlist_local else int(best_idx_local)
                best_position_inner = int(append_position_local)
                best_feat_inner: dict[str, Any] | None = None
                append_best_score_inner = float("-inf")
                append_best_g_lcb_inner = 0.0
                append_best_family_inner = ""
                best_non_append_score_inner = float("-inf")
                best_non_append_g_lcb_inner = 0.0
                records_inner: list[dict[str, Any]] = []
                for idx in shortlist_local:
                    for pos in positions_considered_local:
                        active_window_guess = _predict_reopt_window_for_position(
                            theta=np.asarray(theta_logical_branch, dtype=float),
                            position_id=int(pos),
                            policy=str(adapt_reopt_policy_key),
                            window_size=int(adapt_window_size_val),
                            window_topk=int(adapt_window_topk_val),
                            periodic_full_refit_triggered=False,
                        )
                        inherited_window_guess = [
                            int(i) for i in active_window_guess if int(i) != int(pos)
                        ]
                        compile_est = phase1_compile_oracle.estimate(
                            candidate_term_count=int(len(pool_compiled[int(idx)].terms)),
                            position_id=int(pos),
                            append_position=int(append_position_local),
                            refit_active_count=int(len(inherited_window_guess)),
                        )
                        meas_stats = branch.phase1_measure_cache.estimate([str(pool[int(idx)].label)])
                        is_residual_candidate = bool(int(idx) in phase1_residual_indices)
                        stage_gate_open = (
                            str(stage_name_local) == "residual"
                            or (not bool(is_residual_candidate))
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
                        metric_raw = candidate_metric_cache_local.get(int(idx))
                        if metric_raw is None:
                            apsi_metric = _apply_compiled_polynomial(
                                np.asarray(psi_current_local, dtype=complex),
                                pool_compiled[int(idx)],
                            )
                            mean_metric = complex(
                                np.vdot(np.asarray(psi_current_local, dtype=complex), apsi_metric)
                            )
                            centered_metric = np.asarray(
                                apsi_metric - mean_metric * np.asarray(psi_current_local, dtype=complex),
                                dtype=complex,
                            )
                            metric_raw = float(
                                max(0.0, np.real(np.vdot(centered_metric, centered_metric)))
                            )
                            candidate_metric_cache_local[int(idx)] = float(metric_raw)
                        family_id = str(pool_family_ids[int(idx)])
                        family_repeat_cost = family_repeat_cache_local.get(family_id)
                        if family_repeat_cost is None:
                            family_repeat_cost = float(
                                family_repeat_cost_from_history(
                                    history_rows=list(branch.history),
                                    candidate_family=str(family_id),
                                )
                            )
                            family_repeat_cache_local[family_id] = float(family_repeat_cost)
                        feat_obj = build_candidate_features(
                            stage_name=str(stage_name_local),
                            candidate_label=str(pool[int(idx)].label),
                            candidate_family=str(family_id),
                            candidate_pool_index=int(idx),
                            position_id=int(pos),
                            append_position=int(append_position_local),
                            positions_considered=[int(x) for x in positions_considered_local],
                            gradient_signed=float(gradients_local[int(idx)]),
                            metric_proxy=float(metric_raw),
                            sigma_hat=0.0,
                            refit_window_indices=[int(i) for i in inherited_window_guess],
                            compile_cost=compile_est,
                            measurement_stats=meas_stats,
                            leakage_penalty=0.0,
                            stage_gate_open=bool(stage_gate_open),
                            leakage_gate_open=True,
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
                                str(phase3_lifetime_cost_mode_key) if phase3_enabled else "off"
                            ),
                            remaining_evaluations_proxy_mode=(
                                "remaining_depth"
                                if phase3_enabled and str(phase3_lifetime_cost_mode_key) != "off"
                                else "none"
                            ),
                            family_repeat_cost=float(family_repeat_cost),
                        )
                        window_terms, window_labels = _window_terms_for_position(
                            selected_ops=list(branch.selected_ops),
                            refit_window_indices=[int(i) for i in inherited_window_guess],
                            position_id=int(pos),
                        )
                        score_val = float(feat_obj.simple_score or float("-inf"))
                        records_inner.append(
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
                        if int(pos) == int(append_position_local) and score_val > append_best_score_inner:
                            append_best_score_inner = float(score_val)
                            append_best_g_lcb_inner = float(feat_obj.g_lcb)
                            append_best_family_inner = str(feat_obj.candidate_family)
                        if int(pos) != int(append_position_local) and score_val > best_non_append_score_inner:
                            best_non_append_score_inner = float(score_val)
                            best_non_append_g_lcb_inner = float(feat_obj.g_lcb)
                        if score_val > best_score_local:
                            best_score_local = float(score_val)
                            best_idx_inner = int(idx)
                            best_position_inner = int(pos)
                            best_feat_inner = dict(feat_obj.__dict__)
                return {
                    "best_score": float(best_score_local),
                    "best_idx": int(best_idx_inner),
                    "best_position": int(best_position_inner),
                    "best_feat": (dict(best_feat_inner) if isinstance(best_feat_inner, dict) else None),
                    "append_best_score": float(append_best_score_inner),
                    "append_best_g_lcb": float(append_best_g_lcb_inner),
                    "append_best_family": str(append_best_family_inner),
                    "best_non_append_score": float(best_non_append_score_inner),
                    "best_non_append_g_lcb": float(best_non_append_g_lcb_inner),
                    "records": list(records_inner),
                }

            append_eval_local = _evaluate_phase1_positions_local(
                [int(append_position_local)],
                trough_probe_triggered_local=False,
            )
            score_eval_local = append_eval_local
            best_feat_local = score_eval_local["best_feat"]
            best_idx_local = int(score_eval_local["best_idx"])
            selected_position_local = int(score_eval_local["best_position"])
            phase1_last_selected_score_local = float(score_eval_local["best_score"])
            controller_snapshot_local = None

            if phase2_enabled:
                cheap_records = shortlist_records(
                    [
                        {
                            **dict(rec),
                            "feature": rec["feature"],
                            "simple_score": float(rec.get("simple_score", float("-inf"))),
                            "candidate_pool_index": int(rec.get("candidate_pool_index", -1)),
                            "position_id": int(rec.get("position_id", append_position_local)),
                        }
                        for rec in score_eval_local.get("records", [])
                    ],
                    cfg=phase2_score_cfg,
                    score_key="simple_score",
                )
                phase1_last_retained_records_local = _candidate_feature_rows(cheap_records)
                full_records: list[dict[str, Any]] = []
                phase2_scaffold_context_cache: dict[tuple[int, ...], Any] = {}
                for rec in cheap_records:
                    feat_base = rec.get("feature")
                    if not isinstance(feat_base, CandidateFeatures):
                        continue
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

                    def _full_record_for_candidate_local(
                        *,
                        candidate_term: AnsatzTerm,
                        candidate_label: str,
                        generator_metadata: Mapping[str, Any] | None,
                        symmetry_spec_candidate: Mapping[str, Any] | None,
                        runtime_split_mode_value: str = "off",
                        runtime_split_parent_label_value: str | None = None,
                        runtime_split_child_index_value: int | None = None,
                        runtime_split_child_count_value: int | None = None,
                        runtime_split_chosen_representation_value: str = "parent",
                        runtime_split_child_indices_value: Sequence[int] | None = None,
                        runtime_split_child_labels_value: Sequence[str] | None = None,
                        runtime_split_child_generator_ids_value: Sequence[str] | None = None,
                    ) -> dict[str, Any]:
                        compiled_candidate = phase2_compiled_term_cache.get(str(candidate_label))
                        if compiled_candidate is None:
                            compiled_candidate = _compile_polynomial_action(
                                candidate_term.polynomial,
                                pauli_action_cache=pauli_action_cache,
                            )
                            phase2_compiled_term_cache[str(candidate_label)] = compiled_candidate
                        apsi_candidate = _apply_compiled_polynomial(
                            np.asarray(psi_current_local, dtype=complex),
                            compiled_candidate,
                        )
                        grad_candidate = float(
                            adapt_commutator_grad_from_hpsi(
                                hpsi_current_local,
                                apsi_candidate,
                            )
                        )
                        mean_candidate = complex(
                            np.vdot(np.asarray(psi_current_local, dtype=complex), apsi_candidate)
                        )
                        centered_candidate = np.asarray(
                            apsi_candidate - mean_candidate * np.asarray(psi_current_local, dtype=complex),
                            dtype=complex,
                        )
                        metric_candidate = float(
                            max(0.0, np.real(np.vdot(centered_candidate, centered_candidate)))
                        )
                        compile_est_candidate = phase1_compile_oracle.estimate(
                            candidate_term_count=int(len(compiled_candidate.terms)),
                            position_id=int(feat_base.position_id),
                            append_position=int(feat_base.append_position),
                            refit_active_count=int(len(feat_base.refit_window_indices)),
                        )
                        measurement_stats_candidate = branch.phase1_measure_cache.estimate(
                            [str(candidate_label)]
                        )
                        feat_candidate_base = build_candidate_features(
                            stage_name=str(feat_base.stage_name),
                            candidate_label=str(candidate_label),
                            candidate_family=str(feat_base.candidate_family),
                            candidate_pool_index=int(feat_base.candidate_pool_index),
                            position_id=int(feat_base.position_id),
                            append_position=int(feat_base.append_position),
                            positions_considered=[int(x) for x in feat_base.positions_considered],
                            gradient_signed=float(grad_candidate),
                            metric_proxy=float(metric_candidate),
                            sigma_hat=float(feat_base.sigma_hat),
                            refit_window_indices=[int(i) for i in feat_base.refit_window_indices],
                            compile_cost=compile_est_candidate,
                            measurement_stats=measurement_stats_candidate,
                            leakage_penalty=0.0,
                            stage_gate_open=bool(feat_base.stage_gate_open),
                            leakage_gate_open=True,
                            trough_probe_triggered=bool(feat_base.trough_probe_triggered),
                            trough_detected=bool(feat_base.trough_detected),
                            cfg=phase1_score_cfg,
                            generator_metadata=(dict(generator_metadata) if isinstance(generator_metadata, Mapping) else None),
                            symmetry_spec=(dict(symmetry_spec_candidate) if isinstance(symmetry_spec_candidate, Mapping) else None),
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
                            family_repeat_cost=float(feat_base.family_repeat_cost),
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
                                    "runtime_split_chosen_representation": str(
                                        runtime_split_chosen_representation_value
                                    ),
                                    "runtime_split_child_indices": [
                                        int(x)
                                        for x in (
                                            list(runtime_split_child_indices_value)
                                            if runtime_split_child_indices_value is not None
                                            else []
                                        )
                                    ],
                                    "runtime_split_child_labels": [
                                        str(x)
                                        for x in (
                                            list(runtime_split_child_labels_value)
                                            if runtime_split_child_labels_value is not None
                                            else []
                                        )
                                    ],
                                    "runtime_split_child_generator_ids": [
                                        str(x)
                                        for x in (
                                            list(runtime_split_child_generator_ids_value)
                                            if runtime_split_child_generator_ids_value is not None
                                            else []
                                        )
                                    ],
                                }
                            )
                        active_memory = phase2_memory_adapter.select_active(
                            branch.phase2_optimizer_memory,
                            active_indices=list(feat_candidate_base.refit_window_indices),
                            source=f"beam.depth{int(depth + 1)}.window_subset",
                        )
                        scaffold_key = tuple(int(i) for i in feat_candidate_base.refit_window_indices)
                        scaffold_context = phase2_scaffold_context_cache.get(scaffold_key)
                        if scaffold_context is None:
                            scaffold_context = phase2_novelty_oracle.prepare_scaffold_context(
                                selected_ops=list(branch.selected_ops),
                                theta=np.asarray(theta_logical_branch, dtype=float),
                                psi_ref=np.asarray(psi_ref, dtype=complex),
                                psi_state=np.asarray(psi_current_local, dtype=complex),
                                h_compiled=h_compiled,
                                hpsi_state=np.asarray(hpsi_current_local, dtype=complex),
                                refit_window_indices=list(feat_candidate_base.refit_window_indices),
                                pauli_action_cache=pauli_action_cache,
                            )
                            phase2_scaffold_context_cache[scaffold_key] = scaffold_context
                        feat_full = build_full_candidate_features(
                            base_feature=feat_candidate_base,
                            candidate_term=candidate_term,
                            cfg=phase2_score_cfg,
                            novelty_oracle=phase2_novelty_oracle,
                            curvature_oracle=phase2_curvature_oracle,
                            scaffold_context=scaffold_context,
                            h_compiled=h_compiled,
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
                            "phase2_raw_score": float(feat_full.phase2_raw_score or float("-inf")),
                            "full_v2_score": float(feat_full.full_v2_score or float("-inf")),
                            "candidate_pool_index": int(feat_full.candidate_pool_index),
                            "position_id": int(feat_full.position_id),
                            "candidate_term": candidate_term,
                        }

                    parent_record = _full_record_for_candidate_local(
                        candidate_term=rec["candidate_term"],
                        candidate_label=parent_label,
                        generator_metadata=parent_generator_meta,
                        symmetry_spec_candidate=parent_symmetry_spec,
                    )
                    candidate_variants = [dict(parent_record)]
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
                            phase3_runtime_split_summary_local["probed_parent_count"] = int(
                                phase3_runtime_split_summary_local.get("probed_parent_count", 0)
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
                            phase3_runtime_split_summary_local["evaluated_child_count"] = int(
                                phase3_runtime_split_summary_local.get("evaluated_child_count", 0)
                            ) + 1
                            child_symmetry_gate = (
                                dict(child.get("symmetry_gate", {}))
                                if isinstance(child.get("symmetry_gate"), Mapping)
                                else {}
                            )
                            if not bool(child_symmetry_gate.get("passed", True)):
                                phase3_runtime_split_summary_local["rejected_child_count_symmetry"] = int(
                                    phase3_runtime_split_summary_local.get("rejected_child_count_symmetry", 0)
                                ) + 1
                        child_set_candidates = build_runtime_split_child_sets(
                            parent_label=str(parent_label),
                            family_id=str(feat_base.candidate_family),
                            num_sites=int(num_sites),
                            ordering=str(ordering),
                            qpb=int(max(1, qpb)),
                            split_mode=str(phase3_runtime_split_mode_key),
                            children=split_children,
                            parent_generator_metadata=parent_generator_meta,
                            symmetry_spec=parent_symmetry_spec,
                            max_subset_size=3,
                        )
                        phase3_runtime_split_summary_local["admissible_child_set_count"] = int(
                            phase3_runtime_split_summary_local.get("admissible_child_set_count", 0)
                        ) + int(len(child_set_candidates))
                        best_split_record: dict[str, Any] | None = None
                        best_split_payload: dict[str, Any] | None = None
                        best_split_gate_results: dict[str, Any] = {}
                        best_split_child_ids: list[str] = []
                        best_split_score = float("-inf")
                        split_candidate_scores: dict[str, float] = {}
                        admissible_child_subsets: list[list[str]] = []
                        for child_set in child_set_candidates:
                            split_label = str(child_set.get("candidate_label"))
                            split_poly = child_set.get("candidate_polynomial")
                            split_meta = child_set.get("candidate_generator_metadata")
                            if not isinstance(split_poly, PauliPolynomial):
                                continue
                            if not isinstance(split_meta, Mapping):
                                continue
                            pool_generator_registry[str(split_label)] = dict(split_meta)
                            split_record = _full_record_for_candidate_local(
                                candidate_term=AnsatzTerm(label=str(split_label), polynomial=split_poly),
                                candidate_label=str(split_label),
                                generator_metadata=dict(split_meta),
                                symmetry_spec_candidate=(
                                    dict(split_meta.get("symmetry_spec", {}))
                                    if isinstance(split_meta.get("symmetry_spec"), Mapping)
                                    else parent_symmetry_spec
                                ),
                                runtime_split_mode_value=str(phase3_runtime_split_mode_key),
                                runtime_split_parent_label_value=str(parent_label),
                                runtime_split_child_count_value=int(len(split_children)),
                                runtime_split_chosen_representation_value="child_set",
                                runtime_split_child_indices_value=[
                                    int(x) for x in child_set.get("child_indices", [])
                                ],
                                runtime_split_child_labels_value=[
                                    str(x) for x in child_set.get("child_labels", [])
                                ],
                                runtime_split_child_generator_ids_value=[
                                    str(x) for x in child_set.get("child_generator_ids", [])
                                ],
                            )
                            split_score = float(split_record.get("full_v2_score", float("-inf")))
                            split_candidate_scores[str(split_label)] = float(split_score)
                            admissible_child_subsets.append(
                                [str(x) for x in child_set.get("child_labels", [])]
                            )
                            if split_score > best_split_score:
                                best_split_score = float(split_score)
                                best_split_record = dict(split_record)
                                best_split_payload = dict(child_set)
                                best_split_gate_results = (
                                    dict(child_set.get("symmetry_gate", {}))
                                    if isinstance(child_set.get("symmetry_gate"), Mapping)
                                    else {}
                                )
                                best_split_child_ids = [
                                    str(x) for x in child_set.get("child_generator_ids", [])
                                ]
                        if best_split_record is not None:
                            candidate_variants.append(dict(best_split_record))
                            parent_score = float(parent_record.get("full_v2_score", float("-inf")))
                            split_score = float(best_split_record.get("full_v2_score", float("-inf")))
                            split_wins = bool(split_score > parent_score)
                            if split_wins:
                                phase3_runtime_split_summary_local["probe_child_set_count"] = int(
                                    phase3_runtime_split_summary_local.get("probe_child_set_count", 0)
                                ) + 1
                                best_split_choice_reason = "child_set_actual_score_better"
                            else:
                                phase3_runtime_split_summary_local["probe_parent_win_count"] = int(
                                    phase3_runtime_split_summary_local.get("probe_parent_win_count", 0)
                                ) + 1
                                best_split_choice_reason = "parent_actual_score_better"
                            branch.phase3_split_events.append(
                                build_split_event(
                                    parent_generator_id=str(parent_generator_meta.get("generator_id")),
                                    child_generator_ids=list(best_split_child_ids),
                                    reason=f"depth{int(depth + 1)}_shortlist_probe",
                                    split_mode=str(phase3_runtime_split_mode_key),
                                    probe_trigger="phase2_shortlist",
                                    choice_reason=str(best_split_choice_reason),
                                    parent_score=float(parent_score),
                                    child_scores=dict(split_candidate_scores),
                                    admissible_child_subsets=list(admissible_child_subsets),
                                    chosen_representation=("child_set" if split_wins else "parent"),
                                    chosen_child_ids=(list(best_split_child_ids) if split_wins else []),
                                    split_margin=float(split_score - parent_score),
                                    symmetry_gate_results=dict(best_split_gate_results),
                                    compiled_cost_parent=float(
                                        parent_record.get("feature").compile_cost_total
                                    )
                                    if isinstance(parent_record.get("feature"), CandidateFeatures)
                                    else None,
                                    compiled_cost_children=float(
                                        best_split_record.get("feature").compile_cost_total
                                    )
                                    if isinstance(best_split_record.get("feature"), CandidateFeatures)
                                    else None,
                                    insertion_positions=[int(feat_base.position_id)],
                                )
                            )
                        elif split_children:
                            phase3_runtime_split_summary_local["probe_parent_win_count"] = int(
                                phase3_runtime_split_summary_local.get("probe_parent_win_count", 0)
                            ) + 1
                            branch.phase3_split_events.append(
                                build_split_event(
                                    parent_generator_id=str(parent_generator_meta.get("generator_id")),
                                    child_generator_ids=[
                                        str(meta.get("generator_id"))
                                        for meta in (
                                            child.get("child_generator_metadata")
                                            for child in split_children
                                            if isinstance(child.get("child_generator_metadata"), Mapping)
                                        )
                                        if meta.get("generator_id") is not None
                                    ],
                                    reason=f"depth{int(depth + 1)}_shortlist_probe",
                                    split_mode=str(phase3_runtime_split_mode_key),
                                    probe_trigger="phase2_shortlist",
                                    choice_reason="no_admissible_child_set",
                                    parent_score=float(parent_record.get("full_v2_score", float("-inf"))),
                                    child_scores=dict(split_candidate_scores),
                                    admissible_child_subsets=list(admissible_child_subsets),
                                    chosen_representation="parent",
                                    chosen_child_ids=[],
                                    symmetry_gate_results={"admissible_child_set_count": 0},
                                    compiled_cost_parent=float(
                                        parent_record.get("feature").compile_cost_total
                                    )
                                    if isinstance(parent_record.get("feature"), CandidateFeatures)
                                    else None,
                                    compiled_cost_children=None,
                                    insertion_positions=[int(feat_base.position_id)],
                                )
                            )
                    candidate_variants = sorted(candidate_variants, key=_phase2_record_sort_key)
                    if candidate_variants:
                        full_records.append(dict(candidate_variants[0]))
                full_records = _attach_controller_snapshot(
                    sorted(full_records, key=_phase2_record_sort_key),
                    snapshot=controller_snapshot_local,
                )
                phase2_last_shortlist_eval_records_local = [dict(rec) for rec in full_records]
                phase2_shortlisted_records_local = _phase_shortlist_with_legacy_hook(
                    full_records,
                    score_key="phase2_raw_score",
                    threshold=_controller_threshold(controller_snapshot_local, "phase2"),
                    cap=_controller_cap(controller_snapshot_local, "phase2", phase2_score_cfg.shortlist_size),
                    frontier_ratio=float(phase2_score_cfg.phase2_frontier_ratio),
                    tie_break_score_key="cheap_score",
                    shortlist_flag="phase2_shortlisted",
                )
                phase2_last_geometric_shortlist_records_local = _candidate_feature_rows(
                    phase2_shortlisted_records_local
                )
                phase3_shortlisted_records_local = (
                    _phase_shortlist_with_legacy_hook(
                        phase2_shortlisted_records_local,
                        score_key="full_v2_score",
                        threshold=_controller_threshold(controller_snapshot_local, "phase3"),
                        cap=_controller_cap(controller_snapshot_local, "phase3", phase2_score_cfg.shortlist_size),
                        frontier_ratio=float(phase2_score_cfg.phase3_frontier_ratio),
                        tie_break_score_key="phase2_raw_score",
                        shortlist_flag="phase3_shortlisted",
                    )
                    if phase3_enabled
                    else list(phase2_shortlisted_records_local)
                )
                phase2_last_shortlist_records_local = _candidate_feature_rows(full_records)
                admission_source_records_local = (
                    phase3_shortlisted_records_local
                    if phase3_enabled
                    else phase2_shortlisted_records_local
                )
                phase2_last_retained_shortlist_records_local = _candidate_feature_rows(
                    admission_source_records_local
                )
                retained_records_local = [dict(rec) for rec in admission_source_records_local]
                phase3_tie_selection_meta_local: dict[str, Any] = {
                    "active": False,
                    "band_count": 0,
                    "selected_count": int(len(retained_records_local)),
                    "best_score": float("-inf"),
                    "depth_left": int(max(0, int(max_depth) - int(depth + 1))),
                    "late_coordinate": float(int(depth + 1) / max(1, int(max_depth))),
                    "reason": "disabled",
                }
                if phase3_enabled:
                    retained_records_local, phase3_tie_selection_meta_local = _phase3_tie_beam_selection_pool(
                        admission_source_records_local,
                        default_cap=int(max(1, children_cap)),
                        score_key="full_v2_score",
                        score_ratio=float(phase3_tie_beam_score_ratio_val),
                        abs_tol=float(phase3_tie_beam_abs_tol_val),
                        max_branches=int(phase3_tie_beam_max_branches_val),
                        max_late_coordinate=float(phase3_tie_beam_max_late_coordinate_val),
                        min_depth_left=int(phase3_tie_beam_min_depth_left_val),
                        depth_one_based=int(depth + 1),
                        max_depth_local=int(max_depth),
                    )
                if (
                    not bool(phase3_tie_selection_meta_local.get("active", False))
                    and
                    phase3_enabled
                    and bool(phase2_enable_batching)
                    and str(stage_name_local) == "core"
                    and retained_records_local
                ):
                    retained_records_local, batch_summary_local = reduced_plane_batch_select(
                        retained_records_local,
                        cfg=phase2_score_cfg,
                        selected_ops=list(branch.selected_ops),
                        theta=np.asarray(theta_logical_branch, dtype=float),
                        psi_ref=np.asarray(psi_ref, dtype=complex),
                        psi_state=np.asarray(psi_current_local, dtype=complex),
                        h_compiled=h_compiled,
                        novelty_oracle=phase2_novelty_oracle,
                        curvature_oracle=phase2_curvature_oracle,
                        compiled_cache=phase2_compiled_term_cache,
                        pauli_action_cache=pauli_action_cache,
                        tie_break_score_key="phase2_raw_score",
                    )
                    phase2_last_batch_penalty_total_local = float(
                        batch_summary_local.get("additivity_defect", 0.0)
                    )
                    phase2_last_batch_selected_local = bool(len(retained_records_local) > 1)
                phase2_last_admitted_records_local = _candidate_feature_rows(retained_records_local)
                eligible_records_local = [
                    dict(rec)
                    for rec in retained_records_local
                    if float(rec.get("full_v2_score", float("-inf"))) > 0.0
                ]
                if eligible_records_local:
                    top_record = dict(eligible_records_local[0])
                    top_feat = top_record.get("feature")
                    if isinstance(top_feat, CandidateFeatures):
                        phase1_feature_selected_local = dict(top_feat.__dict__)
                        phase1_last_selected_score_local = float(
                            top_feat.selector_score
                            if top_feat.selector_score is not None
                            else (
                                top_feat.full_v2_score
                                if top_feat.full_v2_score is not None
                                else (
                                    top_feat.phase2_raw_score
                                    if top_feat.phase2_raw_score is not None
                                    else top_feat.simple_score or float("-inf")
                                )
                            )
                        )
                        best_idx_local = int(top_feat.candidate_pool_index)
                        selected_position_local = int(top_feat.position_id)
                    selection_mode_local = (
                        (
                            "phase3_rerank_split"
                            if phase3_enabled
                            and isinstance(top_feat, CandidateFeatures)
                            and str(top_feat.runtime_split_mode) != "off"
                            else ("phase3_rerank" if phase3_enabled else "phase2_raw")
                        )
                        + (
                            "_tie_beam"
                            if bool(phase3_tie_selection_meta_local.get("active", False))
                            else ""
                        )
                    )
                    for rec in eligible_records_local:
                        feat_obj = rec.get("feature")
                        feat_row = (
                            dict(feat_obj.__dict__)
                            if isinstance(feat_obj, CandidateFeatures)
                            else None
                        )
                        candidate_term = rec.get("candidate_term")
                        if not isinstance(candidate_term, AnsatzTerm):
                            candidate_term = pool[int(rec.get("candidate_pool_index", best_idx_local))]
                        candidate_plans.append(
                            _BranchExpansionPlan(
                                candidate_pool_index=int(rec.get("candidate_pool_index", best_idx_local)),
                                position_id=int(rec.get("position_id", append_position_local)),
                                selection_mode=str(selection_mode_local),
                                candidate_label=str(candidate_term.label),
                                candidate_term=candidate_term,
                                feature_row=feat_row,
                                init_theta=0.0,
                            )
                        )
                elif best_feat_local is not None:
                    phase1_feature_selected_local = dict(best_feat_local)
                    phase1_last_selected_score_local = float(
                        best_feat_local.get("simple_score", float("-inf"))
                    )
                    best_idx_local = int(best_feat_local.get("candidate_pool_index", best_idx_local))
                    selected_position_local = int(
                        best_feat_local.get("position_id", selected_position_local)
                    )
                    selection_mode_local = "simple_v1_full_zero_fallback"
                    candidate_plans.append(
                        _BranchExpansionPlan(
                            candidate_pool_index=int(best_idx_local),
                            position_id=int(selected_position_local),
                            selection_mode=str(selection_mode_local),
                            candidate_label=str(pool[int(best_idx_local)].label),
                            candidate_term=pool[int(best_idx_local)],
                            feature_row=dict(best_feat_local),
                            init_theta=0.0,
                        )
                    )
            elif best_feat_local is not None:
                phase1_feature_selected_local = dict(best_feat_local)
                phase1_last_selected_score_local = float(
                    best_feat_local.get("simple_score", float("-inf"))
                )
                simple_records = sorted(
                    list(phase1_shortlist_records_local),
                    key=lambda rec: (
                        -float(rec.get("simple_score", float("-inf"))),
                        int(rec.get("candidate_pool_index", -1)),
                        int(rec.get("position_id", -1)),
                    ),
                )
                for rec in simple_records[: int(max(1, children_cap))]:
                    feat_obj = rec.get("feature")
                    feat_row = (
                        dict(feat_obj.__dict__)
                        if isinstance(feat_obj, CandidateFeatures)
                        else None
                    )
                    candidate_plans.append(
                        _BranchExpansionPlan(
                            candidate_pool_index=int(rec.get("candidate_pool_index", best_idx_local)),
                            position_id=int(rec.get("position_id", append_position_local)),
                            selection_mode="simple_v1",
                            candidate_label=str(rec.get("candidate_term").label),
                            candidate_term=rec.get("candidate_term"),
                            feature_row=feat_row,
                            init_theta=0.0,
                        )
                    )

            phase1_stage_now_local, phase1_stage_transition_reason_local = (
                phase1_stage_local.resolve_stage_transition(
                    drop_plateau_hits=int(branch.drop_plateau_hits),
                    trough_detected=bool(phase1_last_trough_detected_local),
                    residual_opened=bool(phase1_residual_opened_local),
                )
            )
            if (
                phase1_stage_now_local == "residual"
                and (not phase1_residual_opened_local)
                and len(phase1_residual_indices) > 0
            ):
                phase1_residual_opened_local = True
                available_indices_local |= set(int(i) for i in phase1_residual_indices)
                phase1_stage_events_local.append(
                    {
                        "depth": int(depth + 1),
                        "stage_name": "residual",
                        "reason": str(phase1_stage_transition_reason_local),
                    }
                )

            if max_grad_local < float(eps_grad):
                if bool(finite_angle_fallback) and available_indices_local:
                    fallback_scan_size_local = int(len(available_indices_local))
                    best_probe_energy = float(energy_current_local)
                    best_probe_idx: int | None = None
                    best_probe_theta_value: float | None = None
                    fallback_executor_cache: dict[int, CompiledAnsatzExecutor] = {}
                    for idx in available_indices_local:
                        for trial_theta in (float(finite_angle), -float(finite_angle)):
                            trial_ops, trial_theta_vec = _splice_candidate_at_position(
                                ops=list(branch.selected_ops),
                                theta=np.asarray(branch.theta, dtype=float),
                                op=pool[int(idx)],
                                position_id=int(append_position_local),
                                init_theta=float(trial_theta),
                            )
                            if adapt_state_backend_key == "compiled":
                                trial_executor = fallback_executor_cache.get(int(idx))
                                if trial_executor is None:
                                    trial_executor = _build_compiled_executor(trial_ops)
                                    fallback_executor_cache[int(idx)] = trial_executor
                                psi_trial = _prepare_selected_state(
                                    ops_now=trial_ops,
                                    theta_now=trial_theta_vec,
                                    executor_now=trial_executor,
                                    parameter_layout_now=_build_selected_layout(trial_ops),
                                )
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
                            beam_nfev_total += 1
                            if probe_energy < best_probe_energy:
                                best_probe_energy = float(probe_energy)
                                best_probe_idx = int(idx)
                                best_probe_theta_value = float(trial_theta)
                    fallback_best_probe_delta_e_local = float(best_probe_energy - energy_current_local)
                    fallback_best_probe_theta_local = (
                        None if best_probe_theta_value is None else float(best_probe_theta_value)
                    )
                    if (
                        best_probe_idx is not None
                        and (energy_current_local - best_probe_energy)
                        > float(finite_angle_min_improvement)
                    ):
                        best_idx_local = int(best_probe_idx)
                        selected_position_local = int(append_position_local)
                        phase1_feature_selected_local = None
                        phase1_last_selected_score_local = None
                        phase2_last_shortlist_records_local = []
                        phase2_last_shortlist_eval_records_local = []
                        selection_mode_local = "finite_angle_fallback"
                        candidate_plans = [
                            _BranchExpansionPlan(
                                candidate_pool_index=int(best_idx_local),
                                position_id=int(selected_position_local),
                                selection_mode=str(selection_mode_local),
                                candidate_label=str(pool[int(best_idx_local)].label),
                                candidate_term=pool[int(best_idx_local)],
                                feature_row=None,
                                init_theta=float(best_probe_theta_value),
                            )
                        ]
                    else:
                        return _BranchStepScratch(
                            energy_current=energy_current_local,
                            psi_current=np.asarray(psi_current_local, dtype=complex),
                            hpsi_current=np.asarray(hpsi_current_local, dtype=complex),
                            gradients=np.asarray(gradients_local, dtype=float),
                            grad_magnitudes=np.asarray(grad_magnitudes_local, dtype=float),
                            max_grad=float(max_grad_local),
                            gradient_eval_elapsed_s=float(gradient_eval_elapsed_s_local),
                            append_position=int(append_position_local),
                            best_idx=int(best_idx_local),
                            selected_position=int(selected_position_local),
                            selection_mode="eps_grad",
                            stage_name=str(stage_name_local),
                            phase1_feature_selected=phase1_feature_selected_local,
                            phase1_stage_transition_reason=str(phase1_stage_transition_reason_local),
                            phase1_stage_now=str(phase1_stage_now_local),
                            phase1_stage_after_transition=phase1_stage_local,
                            phase1_last_probe_reason=str(phase1_last_probe_reason_local),
                            phase1_last_positions_considered=list(phase1_last_positions_considered_local),
                            phase1_last_trough_detected=bool(phase1_last_trough_detected_local),
                            phase1_last_trough_probe_triggered=bool(phase1_last_trough_probe_triggered_local),
                            phase1_last_selected_score=phase1_last_selected_score_local,
                            phase1_last_retained_records=[dict(x) for x in phase1_last_retained_records_local],
                            phase2_last_shortlist_records=[dict(x) for x in phase2_last_shortlist_records_local],
                            phase2_last_geometric_shortlist_records=[
                                dict(x) for x in phase2_last_geometric_shortlist_records_local
                            ],
                            phase2_last_retained_shortlist_records=[
                                dict(x) for x in phase2_last_retained_shortlist_records_local
                            ],
                            phase2_last_admitted_records=[dict(x) for x in phase2_last_admitted_records_local],
                            phase2_last_batch_selected=bool(phase2_last_batch_selected_local),
                            phase2_last_batch_penalty_total=float(phase2_last_batch_penalty_total_local),
                            phase2_last_optimizer_memory_reused=False,
                            phase2_last_optimizer_memory_source="unavailable",
                            phase2_last_shortlist_eval_records=[dict(x) for x in phase2_last_shortlist_eval_records_local],
                            phase1_residual_opened=bool(phase1_residual_opened_local),
                            available_indices_after_transition=set(available_indices_local),
                            phase1_stage_events_after_transition=[dict(x) for x in phase1_stage_events_local],
                            phase3_runtime_split_summary_after_eval=copy.deepcopy(phase3_runtime_split_summary_local),
                            proposals=[],
                            stop_reason="eps_grad",
                            fallback_scan_size=int(fallback_scan_size_local),
                            fallback_best_probe_delta_e=fallback_best_probe_delta_e_local,
                            fallback_best_probe_theta=fallback_best_probe_theta_local,
                        )
                else:
                    return _BranchStepScratch(
                        energy_current=energy_current_local,
                        psi_current=np.asarray(psi_current_local, dtype=complex),
                        hpsi_current=np.asarray(hpsi_current_local, dtype=complex),
                        gradients=np.asarray(gradients_local, dtype=float),
                        grad_magnitudes=np.asarray(grad_magnitudes_local, dtype=float),
                        max_grad=float(max_grad_local),
                        gradient_eval_elapsed_s=float(gradient_eval_elapsed_s_local),
                        append_position=int(append_position_local),
                        best_idx=int(best_idx_local),
                        selected_position=int(selected_position_local),
                        selection_mode="eps_grad",
                        stage_name=str(stage_name_local),
                        phase1_feature_selected=phase1_feature_selected_local,
                        phase1_stage_transition_reason=str(phase1_stage_transition_reason_local),
                        phase1_stage_now=str(phase1_stage_now_local),
                        phase1_stage_after_transition=phase1_stage_local,
                        phase1_last_probe_reason=str(phase1_last_probe_reason_local),
                        phase1_last_positions_considered=list(phase1_last_positions_considered_local),
                        phase1_last_trough_detected=bool(phase1_last_trough_detected_local),
                        phase1_last_trough_probe_triggered=bool(phase1_last_trough_probe_triggered_local),
                        phase1_last_selected_score=phase1_last_selected_score_local,
                        phase1_last_retained_records=[dict(x) for x in phase1_last_retained_records_local],
                        phase2_last_shortlist_records=[dict(x) for x in phase2_last_shortlist_records_local],
                        phase2_last_geometric_shortlist_records=[
                            dict(x) for x in phase2_last_geometric_shortlist_records_local
                        ],
                        phase2_last_retained_shortlist_records=[
                            dict(x) for x in phase2_last_retained_shortlist_records_local
                        ],
                        phase2_last_admitted_records=[dict(x) for x in phase2_last_admitted_records_local],
                        phase2_last_batch_selected=bool(phase2_last_batch_selected_local),
                        phase2_last_batch_penalty_total=float(phase2_last_batch_penalty_total_local),
                        phase2_last_optimizer_memory_reused=False,
                        phase2_last_optimizer_memory_source="unavailable",
                        phase2_last_shortlist_eval_records=[dict(x) for x in phase2_last_shortlist_eval_records_local],
                        phase1_residual_opened=bool(phase1_residual_opened_local),
                        available_indices_after_transition=set(available_indices_local),
                        phase1_stage_events_after_transition=[dict(x) for x in phase1_stage_events_local],
                        phase3_runtime_split_summary_after_eval=copy.deepcopy(phase3_runtime_split_summary_local),
                        proposals=[],
                        stop_reason="eps_grad",
                        fallback_scan_size=0,
                        fallback_best_probe_delta_e=None,
                        fallback_best_probe_theta=None,
                    )

            if _selector_debug_enabled_for_depth(int(depth + 1)):
                _ai_log(
                    "hardcoded_adapt_phase3_selector_debug",
                    **_selector_debug_payload(
                        depth_one_based=int(depth + 1),
                        beam_enabled=True,
                        selection_mode_value=str(selection_mode_local),
                        stage_name_value=str(stage_name_local),
                        selected_feature_row=phase1_feature_selected_local,
                        scored_rows=phase2_last_shortlist_eval_records_local,
                        phase2_rows=phase2_shortlisted_records_local,
                        phase3_rows=phase3_shortlisted_records_local,
                        admitted_rows=phase2_last_admitted_records_local,
                        split_summary=phase3_runtime_split_summary_local,
                    ),
                )

            return _BranchStepScratch(
                energy_current=energy_current_local,
                psi_current=np.asarray(psi_current_local, dtype=complex),
                hpsi_current=np.asarray(hpsi_current_local, dtype=complex),
                gradients=np.asarray(gradients_local, dtype=float),
                grad_magnitudes=np.asarray(grad_magnitudes_local, dtype=float),
                max_grad=float(max_grad_local),
                gradient_eval_elapsed_s=float(gradient_eval_elapsed_s_local),
                append_position=int(append_position_local),
                best_idx=int(best_idx_local),
                selected_position=int(selected_position_local),
                selection_mode=str(selection_mode_local),
                stage_name=str(stage_name_local),
                phase1_feature_selected=(
                    None if phase1_feature_selected_local is None else dict(phase1_feature_selected_local)
                ),
                phase1_stage_transition_reason=str(phase1_stage_transition_reason_local),
                phase1_stage_now=str(phase1_stage_now_local),
                phase1_stage_after_transition=phase1_stage_local,
                phase1_last_probe_reason=str(phase1_last_probe_reason_local),
                phase1_last_positions_considered=list(phase1_last_positions_considered_local),
                phase1_last_trough_detected=bool(phase1_last_trough_detected_local),
                phase1_last_trough_probe_triggered=bool(phase1_last_trough_probe_triggered_local),
                phase1_last_selected_score=phase1_last_selected_score_local,
                phase1_last_retained_records=[dict(x) for x in phase1_last_retained_records_local],
                phase2_last_shortlist_records=[dict(x) for x in phase2_last_shortlist_records_local],
                phase2_last_geometric_shortlist_records=[
                    dict(x) for x in phase2_last_geometric_shortlist_records_local
                ],
                phase2_last_retained_shortlist_records=[
                    dict(x) for x in phase2_last_retained_shortlist_records_local
                ],
                phase2_last_admitted_records=[dict(x) for x in phase2_last_admitted_records_local],
                phase2_last_batch_selected=bool(phase2_last_batch_selected_local),
                phase2_last_batch_penalty_total=float(phase2_last_batch_penalty_total_local),
                phase2_last_optimizer_memory_reused=False,
                phase2_last_optimizer_memory_source="unavailable",
                phase2_last_shortlist_eval_records=[dict(x) for x in phase2_last_shortlist_eval_records_local],
                phase1_residual_opened=bool(phase1_residual_opened_local),
                available_indices_after_transition=set(available_indices_local),
                phase1_stage_events_after_transition=[dict(x) for x in phase1_stage_events_local],
                phase3_runtime_split_summary_after_eval=copy.deepcopy(phase3_runtime_split_summary_local),
                proposals=list(candidate_plans),
                stop_reason=None,
                fallback_scan_size=int(fallback_scan_size_local),
                fallback_best_probe_delta_e=fallback_best_probe_delta_e_local,
                fallback_best_probe_theta=fallback_best_probe_theta_local,
            )

        def _materialize_beam_child(
            base_branch: _BeamBranchState,
            scratch: _BranchStepScratch,
            plan: _BranchExpansionPlan,
            *,
            depth: int,
            branch_id: int,
        ) -> _BeamBranchState:
            nonlocal beam_nfev_total

            child = _beam_clone_branch(
                base_branch,
                branch_id=int(branch_id),
                parent_branch_id=int(base_branch.branch_id),
            )
            best_idx_local = int(plan.candidate_pool_index)
            selected_position_local = int(plan.position_id)
            selection_mode_local = str(plan.selection_mode)
            phase1_feature_selected_local = (
                None if plan.feature_row is None else dict(plan.feature_row)
            )
            selected_primary_term = plan.candidate_term

            child_layout_before = _build_selected_layout(list(child.selected_ops))
            admitted_layout = _build_selected_layout([selected_primary_term])
            runtime_insert_pos_local = int(
                runtime_insert_position(child_layout_before, int(selected_position_local))
            )
            if phase2_enabled:
                child.phase2_optimizer_memory = phase2_memory_adapter.remap_insert(
                    child.phase2_optimizer_memory,
                    position_id=int(runtime_insert_pos_local),
                    count=int(admitted_layout.runtime_parameter_count),
                )
            child.selected_ops, child.theta = _splice_candidate_at_position(
                ops=list(child.selected_ops),
                theta=np.asarray(child.theta, dtype=float),
                op=selected_primary_term,
                position_id=int(selected_position_local),
                init_theta=float(plan.init_theta),
            )
            child.selection_counts[int(best_idx_local)] += 1
            if not allow_repeats:
                child.available_indices.discard(int(best_idx_local))
            if (
                phase3_enabled
                and isinstance(phase1_feature_selected_local, Mapping)
                and str(phase1_feature_selected_local.get("runtime_split_mode", "off")) != "off"
                and phase1_feature_selected_local.get("parent_generator_id") is not None
            ):
                selected_child_generator_ids = [
                    str(x)
                    for x in phase1_feature_selected_local.get("runtime_split_child_generator_ids", [])
                ]
                selected_child_labels = [
                    str(x)
                    for x in phase1_feature_selected_local.get("runtime_split_child_labels", [])
                ]
                chosen_representation_local = str(
                    phase1_feature_selected_local.get("runtime_split_chosen_representation", "parent")
                )
                child.phase3_split_events.append(
                    build_split_event(
                        parent_generator_id=str(phase1_feature_selected_local.get("parent_generator_id")),
                        child_generator_ids=(
                            list(selected_child_generator_ids)
                            if selected_child_generator_ids
                            else (
                                [str(phase1_feature_selected_local.get("generator_id"))]
                                if phase1_feature_selected_local.get("generator_id") is not None
                                else []
                            )
                        ),
                        reason=f"depth{int(depth + 1)}_selected",
                        split_mode=str(phase1_feature_selected_local.get("runtime_split_mode")),
                        choice_reason="selected_for_admission",
                        chosen_representation=str(chosen_representation_local),
                        chosen_child_ids=list(selected_child_generator_ids),
                        insertion_positions=[int(selected_position_local)],
                    )
                )
                if str(chosen_representation_local) == "child_set":
                    child.phase3_runtime_split_summary["selected_child_set_count"] = int(
                        child.phase3_runtime_split_summary.get("selected_child_set_count", 0)
                    ) + 1
                child.phase3_runtime_split_summary["selected_child_count"] = int(
                    child.phase3_runtime_split_summary.get("selected_child_count", 0)
                ) + int(
                    max(
                        1,
                        len(selected_child_generator_ids)
                        if selected_child_generator_ids
                        else len(selected_child_labels),
                    )
                )
                child.phase3_runtime_split_summary["selected_child_labels"] = [
                    *list(child.phase3_runtime_split_summary.get("selected_child_labels", [])),
                    *(
                        list(selected_child_labels)
                        if selected_child_labels
                        else [str(selected_primary_term.label)]
                    ),
                ]

            energy_prev_local = float(scratch.energy_current)
            theta_before_opt_local = np.array(child.theta, copy=True)
            optimizer_t0_local = time.perf_counter()
            executor_local = _get_beam_executor(child.selected_ops)
            child_layout_after = _build_selected_layout(child.selected_ops)

            def _obj_child(x: np.ndarray) -> float:
                return _evaluate_selected_energy_objective(
                    ops_now=list(child.selected_ops),
                    theta_now=np.asarray(x, dtype=float),
                    executor_now=executor_local,
                    parameter_layout_now=child_layout_after,
                    objective_stage="beam_local_reopt",
                    depth_marker=int(depth + 1),
                )

            theta_logical_child = _logical_theta_alias(
                np.asarray(child.theta, dtype=float),
                child_layout_after,
            )
            n_theta_local = int(theta_logical_child.size)
            depth_local = int(depth + 1)
            depth_cumulative_local = int(adapt_ref_base_depth) + int(depth_local)
            periodic_full_refit_triggered_local = bool(
                adapt_reopt_policy_key == "windowed"
                and adapt_full_refit_every_val > 0
                and depth_cumulative_local % adapt_full_refit_every_val == 0
            )
            reopt_active_indices_local, reopt_policy_effective_local = _resolve_reopt_active_indices(
                policy=adapt_reopt_policy_key,
                n=n_theta_local,
                theta=np.asarray(theta_logical_child, dtype=float),
                window_size=adapt_window_size_val,
                window_topk=adapt_window_topk_val,
                periodic_full_refit_triggered=periodic_full_refit_triggered_local,
            )
            reopt_runtime_active_indices_local = runtime_indices_for_logical_indices(
                child_layout_after,
                reopt_active_indices_local,
            )
            inherited_reopt_indices_local = [
                int(i)
                for i in reopt_active_indices_local
                if int(i) != int(selected_position_local)
            ]
            if isinstance(phase1_feature_selected_local, dict):
                phase1_feature_selected_local["refit_window_indices"] = [
                    int(i) for i in inherited_reopt_indices_local
                ]
            _obj_opt_local, opt_x0_local = _make_reduced_objective(
                np.asarray(child.theta, dtype=float),
                reopt_runtime_active_indices_local,
                _obj_child,
            )
            phase2_active_memory_local = None
            if phase2_enabled and adapt_inner_optimizer_key == "SPSA":
                phase2_active_memory_local = phase2_memory_adapter.select_active(
                    child.phase2_optimizer_memory,
                    active_indices=list(reopt_runtime_active_indices_local),
                    source=f"beam.depth{int(depth + 1)}.opt_active",
                )
                child.phase2_last_optimizer_memory_reused = bool(
                    phase2_active_memory_local.get("reused", False)
                )
                child.phase2_last_optimizer_memory_source = str(
                    phase2_active_memory_local.get("source", "unavailable")
                )
            else:
                child.phase2_last_optimizer_memory_reused = False
                child.phase2_last_optimizer_memory_source = "unavailable"

            if adapt_inner_optimizer_key == "SPSA":
                local_last_hb_t = optimizer_t0_local

                def _beam_local_spsa_callback(ev: dict[str, Any]) -> None:
                    nonlocal local_last_hb_t
                    now = time.perf_counter()
                    if (now - local_last_hb_t) < float(adapt_spsa_progress_every_s):
                        return
                    local_best = float(ev.get("best_fun", float("nan")))
                    _ai_log(
                        "hardcoded_adapt_spsa_heartbeat",
                        stage="beam_local_reopt",
                        depth=int(depth + 1),
                        branch_id=int(child.branch_id),
                        parent_branch_id=(
                            None
                            if child.parent_branch_id is None
                            else int(child.parent_branch_id)
                        ),
                        selected_position=int(selected_position_local),
                        selected_label=str(selected_primary_term.label),
                        iter=int(ev.get("iter", 0)),
                        nfev_opt_so_far=int(ev.get("nfev_so_far", 0)),
                        best_fun=local_best,
                        delta_abs_best=(
                            float(abs(local_best - exact_gs)) if math.isfinite(local_best) else None
                        ),
                        elapsed_opt_s=float(now - optimizer_t0_local),
                    )
                    local_last_hb_t = now

                result_local = spsa_minimize(
                    fun=_obj_opt_local,
                    x0=opt_x0_local,
                    maxiter=int(maxiter),
                    seed=_branch_optimizer_seed(
                        base_seed=int(seed),
                        stage_tag="local_reopt",
                        depth_local=int(depth_local),
                        parent_state_fingerprint=_branch_state_fingerprint(base_branch),
                        proposal_fingerprint=_proposal_fingerprint(parent=base_branch, plan=plan),
                    ),
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
                    callback=_beam_local_spsa_callback,
                    callback_every=int(adapt_spsa_callback_every),
                    memory=(dict(phase2_active_memory_local) if isinstance(phase2_active_memory_local, Mapping) else None),
                    refresh_every=0,
                    precondition_mode=("diag_rms_grad" if phase2_enabled else "none"),
                )
                if len(reopt_active_indices_local) == n_theta_local:
                    child.theta = np.asarray(result_local.x, dtype=float)
                else:
                    result_x_local = np.asarray(result_local.x, dtype=float).ravel()
                    for k, idx in enumerate(reopt_active_indices_local):
                        child.theta[int(idx)] = float(result_x_local[int(k)])
                child.energy_current = float(result_local.fun)
                nfev_opt_local = int(result_local.nfev)
                nit_opt_local = int(result_local.nit)
                opt_success_local = bool(result_local.success)
                opt_message_local = str(result_local.message)
                if phase2_enabled:
                    child.phase2_optimizer_memory = phase2_memory_adapter.merge_active(
                        child.phase2_optimizer_memory,
                        active_indices=list(reopt_active_indices_local),
                        active_state=phase2_memory_adapter.from_result(
                            result_local,
                            method=str(adapt_inner_optimizer_key),
                            parameter_count=int(len(reopt_active_indices_local)),
                            source=f"beam.depth{int(depth + 1)}.spsa_result",
                        ),
                        source=f"beam.depth{int(depth + 1)}.merge",
                    )
            else:
                result_local = _run_scipy_adapt_optimizer(
                    method_key=str(adapt_inner_optimizer_key),
                    objective=_obj_opt_local,
                    x0=opt_x0_local,
                    maxiter=int(maxiter),
                    context_label="ADAPT inner optimizer",
                    scipy_minimize_fn=scipy_minimize,
                )
                if len(reopt_active_indices_local) == n_theta_local:
                    child.theta = np.asarray(result_local.x, dtype=float)
                else:
                    result_x_local = np.asarray(result_local.x, dtype=float).ravel()
                    for k, idx in enumerate(reopt_active_indices_local):
                        child.theta[int(idx)] = float(result_x_local[int(k)])
                child.energy_current = float(result_local.fun)
                nfev_opt_local = int(getattr(result_local, "nfev", 0))
                nit_opt_local = int(getattr(result_local, "nit", 0))
                opt_success_local = bool(getattr(result_local, "success", False))
                opt_message_local = str(getattr(result_local, "message", ""))
                if phase2_enabled:
                    child.phase2_optimizer_memory = phase2_memory_adapter.unavailable(
                        method=str(adapt_inner_optimizer_key),
                        parameter_count=int(child.theta.size),
                        reason="non_spsa_depth_opt",
                    )

            depth_rollback_local = False
            if float(child.energy_current) > float(energy_prev_local):
                child.theta = np.array(theta_before_opt_local, copy=True)
                child.energy_current = float(energy_prev_local)
                depth_rollback_local = True

            optimizer_elapsed_s_local = float(time.perf_counter() - optimizer_t0_local)
            beam_nfev_total += int(nfev_opt_local)
            child.nfev_total_local += int(nfev_opt_local)
            delta_abs_prev_local = float(child.drop_prev_delta_abs)
            delta_abs_current_local = float(abs(float(child.energy_current) - float(exact_gs)))
            delta_abs_drop_local = float(delta_abs_prev_local - delta_abs_current_local)
            child.drop_prev_delta_abs = float(delta_abs_current_local)
            eps_energy_step_abs_local = float(abs(float(child.energy_current) - float(energy_prev_local)))
            eps_energy_low_step_local = bool(eps_energy_step_abs_local < float(eps_energy))
            eps_energy_gate_open_local = bool(
                int(depth_local) >= int(eps_energy_min_extra_depth_effective)
            )
            if eps_energy_gate_open_local:
                if eps_energy_low_step_local:
                    child.eps_energy_low_streak += 1
                else:
                    child.eps_energy_low_streak = 0
            else:
                child.eps_energy_low_streak = 0
            eps_energy_termination_condition_local = bool(eps_energy_gate_open_local) and (
                int(child.eps_energy_low_streak) >= int(eps_energy_patience_effective)
            )
            drop_low_signal_local = None
            drop_low_grad_local = None
            if drop_policy_enabled and int(depth_local) >= int(adapt_drop_min_depth):
                drop_low_signal_local = bool(delta_abs_drop_local < float(adapt_drop_floor))
                if float(adapt_grad_floor) >= 0.0:
                    drop_low_grad_local = bool(
                        float(scratch.max_grad) < float(adapt_grad_floor)
                    )
                else:
                    drop_low_grad_local = True
                if bool(drop_low_signal_local):
                    child.drop_plateau_hits += 1
                else:
                    child.drop_plateau_hits = 0

            selected_grad_signed_value_local = (
                float(phase1_feature_selected_local.get("g_signed"))
                if isinstance(phase1_feature_selected_local, dict)
                and phase1_feature_selected_local.get("g_signed") is not None
                else float(scratch.gradients[int(best_idx_local)])
            )
            selected_grad_abs_value_local = (
                float(phase1_feature_selected_local.get("g_abs"))
                if isinstance(phase1_feature_selected_local, dict)
                and phase1_feature_selected_local.get("g_abs") is not None
                else float(scratch.grad_magnitudes[int(best_idx_local)])
            )
            history_row_local = {
                "depth": int(depth + 1),
                "selected_op": str(selected_primary_term.label),
                "selected_logical_op": str(selected_primary_term.label),
                "selected_logical_size": 1,
                "selected_logical_pool_indices": [int(best_idx_local)],
                "pool_index": int(best_idx_local),
                "selected_ops": [str(selected_primary_term.label)],
                "selected_pool_indices": [int(best_idx_local)],
                "selection_mode": str(selection_mode_local),
                "init_theta": float(plan.init_theta),
                "init_theta_values": [float(plan.init_theta)],
                "max_grad": float(scratch.max_grad),
                "selected_grad_signed": float(selected_grad_signed_value_local),
                "selected_grad_abs": float(selected_grad_abs_value_local),
                "selected_logical_grad_abs": float(selected_grad_abs_value_local),
                "selected_grad_signed_components": [float(selected_grad_signed_value_local)],
                "selected_grad_abs_components": [float(selected_grad_abs_value_local)],
                "parameterization": "single_term",
                "fallback_scan_size": int(scratch.fallback_scan_size),
                "fallback_best_probe_delta_e": (
                    None if scratch.fallback_best_probe_delta_e is None else float(scratch.fallback_best_probe_delta_e)
                ),
                "fallback_best_probe_theta": (
                    None if scratch.fallback_best_probe_theta is None else float(scratch.fallback_best_probe_theta)
                ),
                "fallback_best_probe_theta_values": (
                    None
                    if scratch.fallback_best_probe_theta is None
                    else [float(scratch.fallback_best_probe_theta)]
                ),
                "energy_before_opt": float(energy_prev_local),
                "energy_after_opt": float(child.energy_current),
                "delta_energy": float(child.energy_current - energy_prev_local),
                "delta_abs_prev": float(delta_abs_prev_local),
                "delta_abs_current": float(delta_abs_current_local),
                "delta_abs_drop_from_prev": float(delta_abs_drop_local),
                "opt_method": str(adapt_inner_optimizer_key),
                "reopt_policy": str(adapt_reopt_policy_key),
                "nfev_opt": int(nfev_opt_local),
                "nit_opt": int(nit_opt_local),
                "opt_success": bool(opt_success_local),
                "opt_message": str(opt_message_local),
                "gradient_eval_elapsed_s": float(scratch.gradient_eval_elapsed_s),
                "optimizer_elapsed_s": float(optimizer_elapsed_s_local),
                "iter_elapsed_s": float(time.perf_counter() - optimizer_t0_local) + float(scratch.gradient_eval_elapsed_s),
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
                "drop_low_signal": drop_low_signal_local,
                "drop_low_grad": drop_low_grad_local,
                "drop_plateau_hits": int(child.drop_plateau_hits),
                "depth_rollback": bool(depth_rollback_local),
                "depth_cumulative": int(depth_cumulative_local),
                "adapt_ref_base_depth": int(adapt_ref_base_depth),
                "eps_energy_step_abs": float(eps_energy_step_abs_local),
                "eps_energy_low_step": bool(eps_energy_low_step_local),
                "eps_energy_low_streak": int(child.eps_energy_low_streak),
                "eps_energy_gate_open": bool(eps_energy_gate_open_local),
                "eps_energy_min_extra_depth_effective": int(eps_energy_min_extra_depth_effective),
                "eps_energy_patience_effective": int(eps_energy_patience_effective),
                "eps_energy_termination_enabled": bool(eps_energy_termination_enabled),
                "eps_energy_termination_condition": bool(eps_energy_termination_condition_local),
                "eps_grad_termination_enabled": bool(eps_grad_termination_enabled),
                "eps_grad_threshold_hit": bool(float(scratch.max_grad) < float(eps_grad)),
                "reopt_policy_effective": str(reopt_policy_effective_local),
                "reopt_active_indices": [int(i) for i in reopt_active_indices_local],
                "reopt_active_count": int(len(reopt_active_indices_local)),
                "reopt_periodic_full_refit_triggered": bool(periodic_full_refit_triggered_local),
                "continuation_mode": str(continuation_mode),
                "candidate_family": str(
                    pool_family_ids[int(best_idx_local)]
                    if int(best_idx_local) < len(pool_family_ids)
                    else "legacy"
                ),
                "stage_name": str(scratch.stage_name),
                "stage_transition_reason": str(scratch.phase1_stage_transition_reason),
                "selected_position": int(selected_position_local),
                "selected_positions": [int(selected_position_local)],
                "selected_feature_rows": (
                    [dict(phase1_feature_selected_local)]
                    if isinstance(phase1_feature_selected_local, Mapping)
                    else []
                ),
                "batch_selected": False,
                "batch_size": 1,
                "beam_structural_mode": "stop_or_single_admission",
                "beam_parent_proposal_family_size": int(len(scratch.proposals)),
                "beam_parent_stop_terminal_also_materialized": True,
                "selector_score": _selector_score_value(phase1_feature_selected_local),
                "selector_burden": _selector_burden_value(phase1_feature_selected_local),
                "positions_considered": [int(x) for x in scratch.phase1_last_positions_considered],
                "score_version": (
                    str(phase1_feature_selected_local.get("score_version"))
                    if isinstance(phase1_feature_selected_local, dict)
                    else None
                ),
                "simple_score": (
                    float(phase1_feature_selected_local.get("simple_score"))
                    if isinstance(phase1_feature_selected_local, dict)
                    and phase1_feature_selected_local.get("simple_score") is not None
                    else None
                ),
                "metric_proxy": (
                    float(phase1_feature_selected_local.get("metric_proxy"))
                    if isinstance(phase1_feature_selected_local, dict)
                    else None
                ),
                "curvature_mode": (
                    str(phase1_feature_selected_local.get("curvature_mode"))
                    if isinstance(phase1_feature_selected_local, dict)
                    else None
                ),
                "novelty_mode": (
                    str(phase1_feature_selected_local.get("novelty_mode"))
                    if isinstance(phase1_feature_selected_local, dict)
                    else None
                ),
                "novelty": (
                    phase1_feature_selected_local.get("novelty")
                    if isinstance(phase1_feature_selected_local, dict)
                    else None
                ),
                "g_lcb": (
                    float(phase1_feature_selected_local.get("g_lcb"))
                    if isinstance(phase1_feature_selected_local, dict)
                    and phase1_feature_selected_local.get("g_lcb") is not None
                    else None
                ),
                "F_raw": (
                    float(phase1_feature_selected_local.get("F_raw"))
                    if isinstance(phase1_feature_selected_local, dict)
                    and phase1_feature_selected_local.get("F_raw") is not None
                    else None
                ),
                "F_red": (
                    float(phase1_feature_selected_local.get("F_red"))
                    if isinstance(phase1_feature_selected_local, dict)
                    and phase1_feature_selected_local.get("F_red") is not None
                    else None
                ),
                "h_eff": (
                    float(phase1_feature_selected_local.get("h_eff"))
                    if isinstance(phase1_feature_selected_local, dict)
                    and phase1_feature_selected_local.get("h_eff") is not None
                    else None
                ),
                "ridge_used": (
                    float(phase1_feature_selected_local.get("ridge_used"))
                    if isinstance(phase1_feature_selected_local, dict)
                    and phase1_feature_selected_local.get("ridge_used") is not None
                    else None
                ),
                "family_repeat_cost": (
                    float(phase1_feature_selected_local.get("family_repeat_cost", 0.0))
                    if isinstance(phase1_feature_selected_local, dict)
                    else None
                ),
                "refit_window_indices": [int(i) for i in reopt_active_indices_local],
                "compile_cost_proxy": (
                    dict(phase1_feature_selected_local.get("compiled_position_cost_proxy", {}))
                    if isinstance(phase1_feature_selected_local, dict)
                    else None
                ),
                "measurement_cache_stats": (
                    dict(phase1_feature_selected_local.get("measurement_cache_stats", {}))
                    if isinstance(phase1_feature_selected_local, dict)
                    else None
                ),
                "actual_fallback_mode": (
                    str(phase1_feature_selected_local.get("actual_fallback_mode"))
                    if isinstance(phase1_feature_selected_local, dict)
                    else None
                ),
                "trough_probe_triggered": bool(scratch.phase1_last_trough_probe_triggered),
                "trough_detected": bool(scratch.phase1_last_trough_detected),
                "phase2_raw_score": (
                    float(phase1_feature_selected_local.get("phase2_raw_score"))
                    if isinstance(phase1_feature_selected_local, dict)
                    and phase1_feature_selected_local.get("phase2_raw_score") is not None
                    else None
                ),
                "full_v2_score": (
                    float(phase1_feature_selected_local.get("full_v2_score"))
                    if isinstance(phase1_feature_selected_local, dict)
                    and phase1_feature_selected_local.get("full_v2_score") is not None
                    else None
                ),
                "shortlist_size": int(len(child.phase2_last_shortlist_records)),
                "shortlisted_records": [dict(x) for x in child.phase2_last_shortlist_records],
                "scored_surface_size": int(len(child.phase2_last_shortlist_records)),
                "scored_surface_records": [dict(x) for x in child.phase2_last_shortlist_records],
                "retained_shortlist_size": int(len(child.phase2_last_retained_shortlist_records)),
                "retained_shortlist_records": [
                    dict(x) for x in child.phase2_last_retained_shortlist_records
                ],
                "admitted_record_count": int(len(child.phase2_last_admitted_records)),
                "admitted_records": [dict(x) for x in child.phase2_last_admitted_records],
                "compatibility_penalty_total": float(child.phase2_last_batch_penalty_total),
                "optimizer_memory_reused": bool(child.phase2_last_optimizer_memory_reused),
                "optimizer_memory_source": str(child.phase2_last_optimizer_memory_source),
                "generator_id": (
                    str(phase1_feature_selected_local.get("generator_id"))
                    if isinstance(phase1_feature_selected_local, dict)
                    and phase1_feature_selected_local.get("generator_id") is not None
                    else None
                ),
                "template_id": (
                    str(phase1_feature_selected_local.get("template_id"))
                    if isinstance(phase1_feature_selected_local, dict)
                    and phase1_feature_selected_local.get("template_id") is not None
                    else None
                ),
                "is_macro_generator": (
                    bool(phase1_feature_selected_local.get("is_macro_generator", False))
                    if isinstance(phase1_feature_selected_local, dict)
                    else None
                ),
                "parent_generator_id": (
                    str(phase1_feature_selected_local.get("parent_generator_id"))
                    if isinstance(phase1_feature_selected_local, dict)
                    and phase1_feature_selected_local.get("parent_generator_id") is not None
                    else None
                ),
                "symmetry_mode": (
                    str(phase1_feature_selected_local.get("symmetry_mode"))
                    if isinstance(phase1_feature_selected_local, dict)
                    else None
                ),
                "symmetry_mitigation_mode": (
                    str(phase1_feature_selected_local.get("symmetry_mitigation_mode"))
                    if isinstance(phase1_feature_selected_local, dict)
                    else str(phase3_symmetry_mitigation_mode_key)
                ),
                "symmetry_spec": (
                    dict(phase1_feature_selected_local.get("symmetry_spec", {}))
                    if isinstance(phase1_feature_selected_local, dict)
                    else None
                ),
                "motif_bonus": (
                    float(phase1_feature_selected_local.get("motif_bonus", 0.0))
                    if isinstance(phase1_feature_selected_local, dict)
                    else 0.0
                ),
                "motif_source": (
                    str(phase1_feature_selected_local.get("motif_source"))
                    if isinstance(phase1_feature_selected_local, dict)
                    else None
                ),
                "motif_metadata": (
                    dict(phase1_feature_selected_local.get("motif_metadata", {}))
                    if isinstance(phase1_feature_selected_local, dict)
                    and isinstance(phase1_feature_selected_local.get("motif_metadata"), Mapping)
                    else None
                ),
                "runtime_split_mode": (
                    str(phase1_feature_selected_local.get("runtime_split_mode", "off"))
                    if isinstance(phase1_feature_selected_local, dict)
                    else "off"
                ),
                "runtime_split_parent_label": (
                    str(phase1_feature_selected_local.get("runtime_split_parent_label"))
                    if isinstance(phase1_feature_selected_local, dict)
                    and phase1_feature_selected_local.get("runtime_split_parent_label") is not None
                    else None
                ),
                "runtime_split_child_index": (
                    int(phase1_feature_selected_local.get("runtime_split_child_index"))
                    if isinstance(phase1_feature_selected_local, dict)
                    and phase1_feature_selected_local.get("runtime_split_child_index") is not None
                    else None
                ),
                "runtime_split_child_count": (
                    int(phase1_feature_selected_local.get("runtime_split_child_count"))
                    if isinstance(phase1_feature_selected_local, dict)
                    and phase1_feature_selected_local.get("runtime_split_child_count") is not None
                    else None
                ),
                "lifetime_cost_mode": (
                    str(phase1_feature_selected_local.get("lifetime_cost_mode"))
                    if isinstance(phase1_feature_selected_local, dict)
                    else str(phase3_lifetime_cost_mode_key)
                ),
                "remaining_evaluations_proxy_mode": (
                    str(phase1_feature_selected_local.get("remaining_evaluations_proxy_mode"))
                    if isinstance(phase1_feature_selected_local, dict)
                    else None
                ),
                "remaining_evaluations_proxy": (
                    float(phase1_feature_selected_local.get("remaining_evaluations_proxy", 0.0))
                    if isinstance(phase1_feature_selected_local, dict)
                    else 0.0
                ),
                "lifetime_weight_components": (
                    dict(phase1_feature_selected_local.get("lifetime_weight_components", {}))
                    if isinstance(phase1_feature_selected_local, dict)
                    else None
                ),
            }
            if adapt_inner_optimizer_key == "SPSA":
                history_row_local["spsa_params"] = dict(adapt_spsa_params)
            child.history.append(history_row_local)
            child.last_transition_kind = "admission_child"
            child.last_admission_record_count = int(len(history_row_local.get("selected_feature_rows", [])))
            child.cumulative_selector_score = float(child.cumulative_selector_score) + float(
                history_row_local.get("selector_score", 0.0)
            )
            child.cumulative_selector_burden = float(child.cumulative_selector_burden) + float(
                history_row_local.get("selector_burden", 0.0)
            )
            child.phase1_stage.record_admission(
                selector_step=int(depth + 1),
                energy_before=float(energy_prev_local),
                energy_after_refit=float(child.energy_current),
            )
            if isinstance(phase1_feature_selected_local, dict):
                child.phase1_features_history.append(dict(phase1_feature_selected_local))
            child.phase1_measure_cache.commit([str(selected_primary_term.label)])
            child.phase1_prune_metadata, child.phase1_prune_first_seen_steps = _transport_prune_metadata_after_admission(
                metadata_rows=child.phase1_prune_metadata,
                labels_added=[str(selected_primary_term.label)],
                positions_added=[int(selected_position_local)],
                feature_rows_added=(
                    [dict(phase1_feature_selected_local)]
                    if isinstance(phase1_feature_selected_local, Mapping)
                    else [{}]
                ),
                selector_step=int(depth + 1),
                first_seen_steps=child.phase1_prune_first_seen_steps,
            )
            prune_controller_snapshot_local = (
                phase1_feature_selected_local.get("controller_snapshot")
                if isinstance(phase1_feature_selected_local, Mapping)
                else None
            )
            child.selected_ops, child.theta, child.energy_current, child.phase2_optimizer_memory, child.phase1_prune_metadata, child.phase1_prune_first_seen_steps, child.phase1_last_prune_summary = _execute_live_mature_prune_pass(
                ops_now=list(child.selected_ops),
                theta_now=np.asarray(child.theta, dtype=float),
                energy_now=float(child.energy_current),
                optimizer_memory_now=dict(child.phase2_optimizer_memory),
                metadata_rows=child.phase1_prune_metadata,
                first_seen_steps=child.phase1_prune_first_seen_steps,
                controller_snapshot=(prune_controller_snapshot_local if isinstance(prune_controller_snapshot_local, Mapping) else None),
                selector_step=int(depth + 1),
                admitted_gain=float(max(0.0, float(energy_prev_local) - float(child.energy_current))),
                history_rows=child.history,
            )
            child.history[-1]["post_admission_prune"] = copy.deepcopy(child.phase1_last_prune_summary)

            if (
                drop_policy_enabled
                and int(depth_local) >= int(adapt_drop_min_depth)
                and int(child.drop_plateau_hits) >= int(adapt_drop_patience)
            ):
                if (not child.phase1_residual_opened) and len(phase1_residual_indices) > 0:
                    child.phase1_residual_opened = True
                    child.available_indices |= set(int(i) for i in phase1_residual_indices)
                    child.phase1_stage.resolve_stage_transition(
                        drop_plateau_hits=int(child.drop_plateau_hits),
                        trough_detected=bool(child.phase1_last_trough_detected),
                        residual_opened=True,
                    )
                    child.drop_plateau_hits = 0
                    child.phase1_stage_events.append(
                        {
                            "depth": int(depth_local),
                            "stage_name": "residual",
                            "reason": "drop_plateau_open",
                        }
                    )
                else:
                    child.terminated = True
                    child.stop_reason = "drop_plateau"
            if (not child.terminated) and bool(eps_energy_termination_condition_local) and bool(
                eps_energy_termination_enabled
            ):
                child.terminated = True
                child.stop_reason = "eps_energy"
            if (not child.terminated) and (not allow_repeats) and (not child.available_indices):
                child.terminated = True
                child.stop_reason = "pool_exhausted"
            if not child.terminated:
                child.stop_reason = None
            child.depth_local = int(depth_local)
            return child

        if bool(beam_policy.beam_enabled):
            root_branch = _BeamBranchState(
                branch_id=0,
                parent_branch_id=None,
                depth_local=0,
                terminated=False,
                stop_reason=None,
                selected_ops=list(selected_ops),
                theta=np.asarray(theta, dtype=float).copy(),
                energy_current=float(energy_current),
                available_indices=set(int(x) for x in available_indices),
                selection_counts=np.asarray(selection_counts, dtype=np.int64).copy(),
                history=[dict(x) for x in history],
                phase1_stage=phase1_stage.clone(),
                phase1_residual_opened=bool(phase1_residual_opened),
                phase1_last_probe_reason=str(phase1_last_probe_reason),
                phase1_last_positions_considered=[int(x) for x in phase1_last_positions_considered],
                phase1_last_trough_detected=bool(phase1_last_trough_detected),
                phase1_last_trough_probe_triggered=bool(phase1_last_trough_probe_triggered),
                phase1_last_selected_score=phase1_last_selected_score,
                phase1_features_history=[dict(x) for x in phase1_features_history],
                phase1_stage_events=[dict(x) for x in phase1_stage_events],
                phase1_measure_cache=phase1_measure_cache.clone(),
                phase1_last_retained_records=[dict(x) for x in phase1_last_retained_records],
                phase2_optimizer_memory=copy.deepcopy(phase2_optimizer_memory),
                phase2_last_shortlist_records=[dict(x) for x in phase2_last_shortlist_records],
                phase2_last_geometric_shortlist_records=[
                    dict(x) for x in phase2_last_geometric_shortlist_records
                ],
                phase2_last_retained_shortlist_records=[
                    dict(x) for x in phase2_last_retained_shortlist_records
                ],
                phase2_last_admitted_records=[dict(x) for x in phase2_last_admitted_records],
                phase2_last_batch_selected=bool(phase2_last_batch_selected),
                phase2_last_batch_penalty_total=float(phase2_last_batch_penalty_total),
                phase2_last_optimizer_memory_reused=bool(phase2_last_optimizer_memory_reused),
                phase2_last_optimizer_memory_source=str(phase2_last_optimizer_memory_source),
                phase2_last_shortlist_eval_records=[dict(x) for x in phase2_last_shortlist_eval_records],
                drop_prev_delta_abs=float(drop_prev_delta_abs),
                drop_plateau_hits=int(drop_plateau_hits),
                eps_energy_low_streak=int(eps_energy_low_streak),
                phase3_split_events=[dict(x) for x in phase3_split_events],
                phase3_runtime_split_summary=copy.deepcopy(phase3_runtime_split_summary),
                phase3_motif_usage=copy.deepcopy(phase3_motif_usage),
                phase3_rescue_history=[dict(x) for x in phase3_rescue_history],
                phase1_prune_metadata=[
                    ScaffoldCoordinateMetadata(**dict(x.__dict__)) for x in phase1_prune_metadata_state
                ],
                phase1_prune_first_seen_steps={
                    str(k): int(v) for k, v in phase1_prune_first_seen_steps.items()
                },
                phase1_last_prune_summary=copy.deepcopy(prune_summary),
                last_transition_kind="root",
                last_admission_record_count=0,
                cumulative_selector_score=float(sum(_selector_score_value(row) for row in history)),
                cumulative_selector_burden=float(sum(_selector_burden_value(row) for row in history)),
                nfev_total_local=int(nfev_total),
            )
            frontier: list[_BeamBranchState] = [root_branch]
            terminals: list[_BeamBranchState] = []
            for depth in range(int(max_depth)):
                if not frontier:
                    break
                child_frontier: list[_BeamBranchState] = []
                round_terminals: list[_BeamBranchState] = []
                round_admission_children: list[_BeamBranchState] = []
                parents_expanded_count = 0
                proposals_selected_count = 0
                proposal_family_count = 0
                stop_children_count = 0
                frontier_input_count = int(len(frontier))
                round_live_cap = int(beam_policy.live_branches_effective)
                for parent in frontier:
                    parents_expanded_count += 1
                    scratch = _evaluate_beam_branch(
                        parent,
                        depth=int(depth),
                        children_cap=int(beam_policy.children_per_parent_effective),
                    )
                    base_branch = _beam_clone_branch(
                        parent,
                        branch_id=int(parent.branch_id),
                        parent_branch_id=parent.parent_branch_id,
                    )
                    base_branch.energy_current = float(scratch.energy_current)
                    base_branch.available_indices = set(
                        int(x) for x in scratch.available_indices_after_transition
                    )
                    base_branch.phase1_stage = scratch.phase1_stage_after_transition.clone()
                    base_branch.phase1_residual_opened = bool(scratch.phase1_residual_opened)
                    base_branch.phase1_stage_events = [
                        dict(x) for x in scratch.phase1_stage_events_after_transition
                    ]
                    base_branch.phase1_last_probe_reason = str(scratch.phase1_last_probe_reason)
                    base_branch.phase1_last_positions_considered = [
                        int(x) for x in scratch.phase1_last_positions_considered
                    ]
                    base_branch.phase1_last_trough_detected = bool(
                        scratch.phase1_last_trough_detected
                    )
                    base_branch.phase1_last_trough_probe_triggered = bool(
                        scratch.phase1_last_trough_probe_triggered
                    )
                    base_branch.phase1_last_selected_score = scratch.phase1_last_selected_score
                    base_branch.phase1_last_retained_records = [
                        dict(x) for x in scratch.phase1_last_retained_records
                    ]
                    base_branch.phase2_last_shortlist_records = [
                        dict(x) for x in scratch.phase2_last_shortlist_records
                    ]
                    base_branch.phase2_last_geometric_shortlist_records = [
                        dict(x) for x in scratch.phase2_last_geometric_shortlist_records
                    ]
                    base_branch.phase2_last_retained_shortlist_records = [
                        dict(x) for x in scratch.phase2_last_retained_shortlist_records
                    ]
                    base_branch.phase2_last_admitted_records = [
                        dict(x) for x in scratch.phase2_last_admitted_records
                    ]
                    base_branch.phase2_last_batch_selected = bool(
                        scratch.phase2_last_batch_selected
                    )
                    base_branch.phase2_last_batch_penalty_total = float(
                        scratch.phase2_last_batch_penalty_total
                    )
                    base_branch.phase2_last_optimizer_memory_reused = bool(
                        scratch.phase2_last_optimizer_memory_reused
                    )
                    base_branch.phase2_last_optimizer_memory_source = str(
                        scratch.phase2_last_optimizer_memory_source
                    )
                    base_branch.phase2_last_shortlist_eval_records = [
                        dict(x) for x in scratch.phase2_last_shortlist_eval_records
                    ]
                    base_branch.phase3_runtime_split_summary = copy.deepcopy(
                        scratch.phase3_runtime_split_summary_after_eval
                    )
                    proposal_family_count += int(len(scratch.proposals))
                    terminal_branch = _beam_clone_branch(
                        base_branch,
                        branch_id=int(base_branch.branch_id),
                        parent_branch_id=base_branch.parent_branch_id,
                    )
                    terminal_branch.last_transition_kind = "stop_child"
                    terminal_branch.last_admission_record_count = 0
                    terminal_branch.terminated = True
                    terminal_branch.stop_reason = (
                        str(scratch.stop_reason)
                        if scratch.stop_reason is not None
                        else ("stop" if scratch.proposals else "empty")
                    )
                    round_terminals.append(terminal_branch)
                    stop_children_count += 1
                    if scratch.stop_reason is not None or not scratch.proposals:
                        continue
                    selected_plans = list(scratch.proposals)
                    proposals_selected_count += int(len(selected_plans))
                    round_live_cap = int(max(int(round_live_cap), int(len(selected_plans))))
                    for plan in selected_plans:
                        child = _materialize_beam_child(
                            base_branch,
                            scratch,
                            plan,
                            depth=int(depth),
                            branch_id=int(beam_branch_counter),
                        )
                        beam_branch_counter += 1
                        round_admission_children.append(child)
                        if child.terminated:
                            round_terminals.append(child)
                        else:
                            child_frontier.append(child)
                frontier_unique = _beam_dedup(child_frontier)
                terminal_candidates = [*terminals, *round_terminals]
                terminal_unique = _beam_dedup(terminal_candidates)
                terminals = _beam_prune(
                    terminal_candidates,
                    cap=int(max(int(beam_policy.terminated_keep_effective), int(round_live_cap))),
                )
                frontier = _beam_prune(
                    child_frontier,
                    cap=int(max(1, int(round_live_cap))),
                )
                round_prune_audits = [
                    _compact_prune_audit(branch.phase1_last_prune_summary)
                    for branch in round_admission_children
                ]
                round_prune_reason_counts: dict[str, int] = {}
                for audit in round_prune_audits:
                    reason_key = str(audit.get("permission_reason", "unknown"))
                    round_prune_reason_counts[reason_key] = int(round_prune_reason_counts.get(reason_key, 0)) + 1
                beam_search_diagnostics["rounds"].append(
                    {
                        "depth": int(depth + 1),
                        "frontier_input_count": int(frontier_input_count),
                        "parents_expanded_count": int(parents_expanded_count),
                        "proposals_selected_count": int(proposals_selected_count),
                        "proposal_family_count": int(proposal_family_count),
                        "stop_children_count": int(stop_children_count),
                        "children_materialized_count": int(len(child_frontier) + len(round_terminals)),
                        "active_children_raw_count": int(len(child_frontier)),
                        "active_children_unique_count": int(len(frontier_unique)),
                        "frontier_kept_count": int(len(frontier)),
                        "frontier_cap_effective": int(max(1, int(round_live_cap))),
                        "round_terminals_raw_count": int(len(round_terminals)),
                        "terminal_pool_candidate_count": int(len(terminal_candidates)),
                        "terminal_pool_unique_count": int(len(terminal_unique)),
                        "terminal_kept_count": int(len(terminals)),
                        "prune_child_count": int(len(round_admission_children)),
                        "prune_permission_open_count": int(sum(1 for audit in round_prune_audits if bool(audit.get("permission_open", False)))),
                        "prune_executed_count": int(sum(1 for audit in round_prune_audits if bool(audit.get("executed", False)))),
                        "prune_accepted_count": int(sum(int(audit.get("accepted_count", 0) or 0) for audit in round_prune_audits)),
                        "prune_permission_reason_counts": dict(round_prune_reason_counts),
                        "prune_audits": [dict(audit) for audit in round_prune_audits],
                    }
                )
            finalists = _beam_dedup([*frontier, *terminals])
            if not finalists:
                finalists = [root_branch]
            for branch in finalists:
                if branch.stop_reason is None:
                    branch.stop_reason = "max_depth"
            winner_branch = sorted(finalists, key=_beam_prune_key)[0]
            beam_search_diagnostics.update(
                {
                    "frontier_final_count": int(len(frontier)),
                    "terminal_final_count": int(len(terminals)),
                    "finalist_count": int(len(finalists)),
                    "winner_branch_id": int(winner_branch.branch_id),
                    "winner_parent_branch_id": (
                        None
                        if winner_branch.parent_branch_id is None
                        else int(winner_branch.parent_branch_id)
                    ),
                    "winner_stop_reason": str(winner_branch.stop_reason or "max_depth"),
                    "winner_fingerprint": str(_branch_state_fingerprint(winner_branch)),
                    "winner_prune_key": dict(_beam_prune_key_payload(winner_branch)),
                    "winner_prune_summary": _compact_prune_audit(winner_branch.phase1_last_prune_summary),
                    "winner_branch_summary": _beam_branch_summary(winner_branch),
                    "winner_branch_state_summary": dict(
                        _beam_branch_summary(winner_branch).get("branch_state_summary", {})
                    ),
                    "winner_optimizer_memory_contract": dict(
                        _beam_branch_summary(winner_branch).get(
                            "optimizer_memory_contract_summary",
                            {},
                        )
                    ),
                    "finalist_summaries": [
                        _beam_branch_summary(branch)
                        for branch in sorted(finalists, key=_beam_prune_key)
                    ],
                }
            )
            selected_ops = list(winner_branch.selected_ops)
            theta = np.asarray(winner_branch.theta, dtype=float).copy()
            selected_layout = _build_selected_layout(selected_ops)
            selected_executor = None
            history = [dict(x) for x in winner_branch.history]
            nfev_total = int(beam_nfev_total)
            stop_reason = str(winner_branch.stop_reason or "max_depth")
            available_indices = set(int(x) for x in winner_branch.available_indices)
            selection_counts = np.asarray(winner_branch.selection_counts, dtype=np.int64).copy()
            phase1_stage = winner_branch.phase1_stage.clone()
            phase1_residual_opened = bool(winner_branch.phase1_residual_opened)
            phase1_last_probe_reason = str(winner_branch.phase1_last_probe_reason)
            phase1_last_positions_considered = [
                int(x) for x in winner_branch.phase1_last_positions_considered
            ]
            phase1_last_trough_detected = bool(winner_branch.phase1_last_trough_detected)
            phase1_last_trough_probe_triggered = bool(
                winner_branch.phase1_last_trough_probe_triggered
            )
            phase1_last_selected_score = winner_branch.phase1_last_selected_score
            phase1_features_history = [dict(x) for x in winner_branch.phase1_features_history]
            phase1_stage_events = [dict(x) for x in winner_branch.phase1_stage_events]
            phase1_measure_cache = winner_branch.phase1_measure_cache.clone()
            phase1_last_retained_records = [dict(x) for x in winner_branch.phase1_last_retained_records]
            phase2_optimizer_memory = copy.deepcopy(winner_branch.phase2_optimizer_memory)
            phase2_last_shortlist_records = [
                dict(x) for x in winner_branch.phase2_last_shortlist_records
            ]
            phase2_last_geometric_shortlist_records = [
                dict(x) for x in winner_branch.phase2_last_geometric_shortlist_records
            ]
            phase2_last_retained_shortlist_records = [
                dict(x) for x in winner_branch.phase2_last_retained_shortlist_records
            ]
            phase2_last_admitted_records = [
                dict(x) for x in winner_branch.phase2_last_admitted_records
            ]
            phase2_last_batch_selected = bool(winner_branch.phase2_last_batch_selected)
            phase2_last_batch_penalty_total = float(
                winner_branch.phase2_last_batch_penalty_total
            )
            phase2_last_optimizer_memory_reused = bool(
                winner_branch.phase2_last_optimizer_memory_reused
            )
            phase2_last_optimizer_memory_source = str(
                winner_branch.phase2_last_optimizer_memory_source
            )
            phase2_last_shortlist_eval_records = [
                dict(x) for x in winner_branch.phase2_last_shortlist_eval_records
            ]
            energy_current = float(winner_branch.energy_current)
            drop_prev_delta_abs = float(winner_branch.drop_prev_delta_abs)
            drop_plateau_hits = int(winner_branch.drop_plateau_hits)
            eps_energy_low_streak = int(winner_branch.eps_energy_low_streak)
            phase3_split_events = [dict(x) for x in winner_branch.phase3_split_events]
            phase3_runtime_split_summary = copy.deepcopy(
                winner_branch.phase3_runtime_split_summary
            )
            phase3_motif_usage = copy.deepcopy(winner_branch.phase3_motif_usage)
            phase3_rescue_history = [dict(x) for x in winner_branch.phase3_rescue_history]
            phase1_prune_metadata_state = [
                ScaffoldCoordinateMetadata(**dict(x.__dict__)) for x in winner_branch.phase1_prune_metadata
            ]
            phase1_prune_first_seen_steps = {
                str(k): int(v) for k, v in winner_branch.phase1_prune_first_seen_steps.items()
            }
            prune_summary = copy.deepcopy(winner_branch.phase1_last_prune_summary)

        for depth in ([] if bool(beam_policy.beam_enabled) else range(int(max_depth))):
            iter_t0 = time.perf_counter()

            # 1) Compute the current state
            if adapt_state_backend_key == "compiled" and len(selected_ops) > 0 and selected_executor is None:
                selected_executor = _build_compiled_executor(selected_ops)
            psi_current = _prepare_selected_state(
                ops_now=selected_ops,
                theta_now=theta,
                executor_now=selected_executor,
                parameter_layout_now=selected_layout,
            )
            energy_current_exact_loop, hpsi_current = energy_via_one_apply(psi_current, h_compiled)
            if not phase3_oracle_inner_objective_enabled:
                energy_current = float(energy_current_exact_loop)
            theta_logical_current = _logical_theta_alias(theta, selected_layout)
            backend_compile_snapshot = (
                backend_compile_oracle.snapshot_base(selected_ops)
                if backend_compile_oracle is not None
                else None
            )
            phase3_base_metric_cache: dict[str, float] = {}
            phase3_sigma_by_label: dict[str, float] = {}

            def _base_metric_for_candidate(
                *,
                candidate_label: str,
                candidate_term: AnsatzTerm,
                gradient_signed: float,
                precompiled_action: Any | None = None,
            ) -> float:
                gradient_abs = float(abs(float(gradient_signed)))
                if not phase3_enabled:
                    return float(gradient_abs)
                cache_key = str(candidate_label)
                cached_value = phase3_base_metric_cache.get(cache_key)
                if cached_value is not None:
                    return float(cached_value)
                if precompiled_action is not None and cache_key not in phase2_compiled_term_cache:
                    phase2_compiled_term_cache[cache_key] = precompiled_action
                try:
                    metric_value = float(
                        raw_f_metric_from_state(
                            psi_state=np.asarray(psi_current, dtype=complex),
                            candidate_label=cache_key,
                            candidate_term=candidate_term,
                            compiled_cache=phase2_compiled_term_cache,
                            pauli_action_cache=pauli_action_cache,
                        )
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to compute phase3 raw F metric for candidate '{cache_key}' at depth {int(depth + 1)}."
                    ) from exc
                phase3_base_metric_cache[cache_key] = float(metric_value)
                return float(metric_value)

            # 2) Compute candidate gradients for all pool operators
            gradient_eval_t0 = time.perf_counter()
            candidate_gradient_scout: list[dict[str, Any]] = []
            gradients = np.zeros(len(pool), dtype=float)
            grad_magnitudes = np.zeros(len(pool), dtype=float)
            if phase3_oracle_gradient_enabled:
                gradients, grad_magnitudes, phase3_sigma_by_label, candidate_gradient_scout, oracle_calls_now = (
                    _phase3_oracle_gradient_scout(
                        selected_ops_now=list(selected_ops),
                        theta_now=np.asarray(theta, dtype=float),
                        append_position_now=int(len(selected_ops)),
                        available_indices_now=sorted(int(i) for i in available_indices),
                    )
                )
                phase3_oracle_gradient_calls_total += int(oracle_calls_now)
                phase3_last_candidate_gradient_scout = [dict(row) for row in candidate_gradient_scout]
                phase3_last_max_gradient_stderr = float(
                    max(
                        (float(row.get("gradient_stderr", 0.0)) for row in candidate_gradient_scout),
                        default=0.0,
                    )
                )
            else:
                for i in available_indices:
                    apsi = _apply_compiled_polynomial(psi_current, pool_compiled[i])
                    if adapt_analytic_noise_enabled:
                        grad_exact = float(adapt_commutator_grad_from_hpsi(hpsi_current, apsi))
                        gradients[i] = _add_adapt_analytic_noise(grad_exact)
                    else:
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
                gradient_source=str(phase3_gradient_source_name),
            )

            # 2b) Select candidate (legacy argmax or phase1_v1 simple score).
            selected_position = int(len(selected_ops))
            stage_name = "legacy"
            phase1_feature_selected: dict[str, Any] | None = None
            phase1_stage_transition_reason = "legacy"
            append_position = int(len(selected_ops))
            phase1_append_best_score = float("-inf")
            phase1_last_retained_records = []
            phase2_selected_records: list[dict[str, Any]] = []
            phase2_last_shortlist_records = []
            phase2_last_geometric_shortlist_records = []
            phase2_last_retained_shortlist_records = []
            phase2_last_admitted_records = []
            phase2_last_shortlist_eval_records = []
            phase2_last_batch_selected = False
            phase2_last_batch_penalty_total = 0.0
            best_logical_idx: int | None = None
            selected_logical_label: str | None = None
            selected_logical_size = 1
            selected_logical_pool_indices: list[int] = []
            selected_grad_signed_components: list[float] = []
            selected_grad_abs_components: list[float] = []
            logical_grad_scores = (
                np.zeros(len(logical_candidates), dtype=float)
                if seq2p_logical_mode
                else np.zeros(0, dtype=float)
            )
            logical_grad_signed_components_all: list[list[float]] = (
                [[] for _ in logical_candidates]
                if seq2p_logical_mode
                else []
            )
            logical_grad_abs_components_all: list[list[float]] = (
                [[] for _ in logical_candidates]
                if seq2p_logical_mode
                else []
            )
            if seq2p_logical_mode:
                for logical_idx, logical_candidate in enumerate(logical_candidates):
                    score, signed_components, abs_components = _logical_candidate_gradient_summary(
                        logical_candidate,
                        gradients,
                    )
                    logical_grad_scores[int(logical_idx)] = float(score)
                    logical_grad_signed_components_all[int(logical_idx)] = [float(x) for x in signed_components]
                    logical_grad_abs_components_all[int(logical_idx)] = [float(x) for x in abs_components]
            if seq2p_logical_mode:
                if logical_available_indices:
                    max_grad = float(max(float(logical_grad_scores[i]) for i in logical_available_indices))
                else:
                    max_grad = 0.0
            elif available_indices:
                max_grad = float(max(float(grad_magnitudes[i]) for i in available_indices))
            else:
                max_grad = 0.0
            if seq2p_logical_mode and (not allow_repeats) and len(logical_available_indices) == 0:
                stop_reason = "pool_exhausted"
                _ai_log(
                    "hardcoded_adapt_pool_exhausted",
                    depth=int(depth + 1),
                    pool_type=str(pool_key),
                    continuation_mode=str(continuation_mode),
                )
                break
            if phase1_enabled and available_indices:
                stage_name = str(phase1_stage.stage_name)
                append_position = int(theta.size)
                def _phase1_cap_key(pool_index: int) -> tuple[float, float, int]:
                    sigma_hat_cap = _phase3_sigma_hat_for_label(
                        candidate_label=str(pool[int(pool_index)].label),
                        sigma_by_label=phase3_sigma_by_label,
                        phase3_enabled=phase3_enabled,
                    )
                    g_lcb_cap = max(
                        float(grad_magnitudes[int(pool_index)])
                        - float(phase1_score_cfg.z_alpha) * float(sigma_hat_cap),
                        0.0,
                    )
                    return (
                        -float(g_lcb_cap),
                        -float(grad_magnitudes[int(pool_index)]),
                        int(pool_index),
                    )

                available_sorted = (
                    sorted(list(available_indices), key=_phase1_cap_key)
                    if phase3_enabled
                    else sorted(list(available_indices), key=lambda i: -float(grad_magnitudes[i]))
                )
                shortlist = available_sorted[: min(len(available_sorted), int(phase1_shortlist_size_val))]
                current_active_window_for_probe, _probe_window_name = _resolve_reopt_active_indices(
                    policy=str(adapt_reopt_policy_key),
                    n=int(max(1, append_position)),
                    theta=(np.asarray(theta_logical_current, dtype=float) if append_position > 0 else np.zeros(1, dtype=float)),
                    window_size=int(adapt_window_size_val),
                    window_topk=int(adapt_window_topk_val),
                    periodic_full_refit_triggered=False,
                )
                candidate_metric_cache: dict[int, float] = {}
                family_repeat_cache: dict[str, float] = {}

                def _cheap_selection_mode(
                    feature_row: Mapping[str, Any] | None,
                    *,
                    probe: bool,
                ) -> str:
                    version = "simple_v1"
                    if isinstance(feature_row, Mapping):
                        version_raw = feature_row.get(
                            "cheap_score_version",
                            feature_row.get("score_version", "simple_v1"),
                        )
                        if version_raw not in {None, ""}:
                            version = str(version_raw)
                    return f"{version}_probe" if probe else str(version)

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
                        candidate_label_local = str(pool[int(idx)].label)
                        candidate_metric_proxy = _base_metric_for_candidate(
                            candidate_label=str(candidate_label_local),
                            candidate_term=pool[int(idx)],
                            gradient_signed=float(gradients[int(idx)]),
                            precompiled_action=pool_compiled[int(idx)],
                        )
                        candidate_sigma_hat = _phase3_sigma_hat_for_label(
                            candidate_label=str(candidate_label_local),
                            sigma_by_label=phase3_sigma_by_label,
                            phase3_enabled=phase3_enabled,
                        )
                        for pos in positions_considered_local:
                            active_window_guess = _predict_reopt_window_for_position(
                                theta=np.asarray(theta_logical_current, dtype=float),
                                position_id=int(pos),
                                policy=str(adapt_reopt_policy_key),
                                window_size=int(adapt_window_size_val),
                                window_topk=int(adapt_window_topk_val),
                                periodic_full_refit_triggered=False,
                            )
                            proxy_compile_est = phase1_compile_oracle.estimate(
                                candidate_term_count=int(len(pool_compiled[int(idx)].terms)),
                                position_id=int(pos),
                                append_position=int(append_position),
                                refit_active_count=int(len(active_window_guess)),
                                candidate_term=pool[int(idx)],
                            )
                            compile_est = (
                                backend_compile_oracle.estimate_insertion(
                                    backend_compile_snapshot,
                                    candidate_term=pool[int(idx)],
                                    position_id=int(pos),
                                    proxy_baseline=proxy_compile_est,
                                )
                                if backend_compile_oracle is not None and backend_compile_snapshot is not None
                                else proxy_compile_est
                            )
                            meas_stats = phase1_measure_cache.estimate(
                                measurement_group_keys_for_term(pool[int(idx)])
                            )
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
                            metric_raw = candidate_metric_cache.get(int(idx))
                            if metric_raw is None:
                                apsi_metric = _apply_compiled_polynomial(
                                    np.asarray(psi_current, dtype=complex),
                                    pool_compiled[int(idx)],
                                )
                                mean_metric = complex(np.vdot(np.asarray(psi_current, dtype=complex), apsi_metric))
                                centered_metric = np.asarray(
                                    apsi_metric - mean_metric * np.asarray(psi_current, dtype=complex),
                                    dtype=complex,
                                )
                                metric_raw = float(max(0.0, np.real(np.vdot(centered_metric, centered_metric))))
                                candidate_metric_cache[int(idx)] = float(metric_raw)
                            family_id = str(pool_family_ids[int(idx)])
                            family_repeat_cost = family_repeat_cache.get(family_id)
                            if family_repeat_cost is None:
                                family_repeat_cost = float(
                                    family_repeat_cost_from_history(
                                        history_rows=list(history),
                                        candidate_family=str(family_id),
                                    )
                                )
                                family_repeat_cache[family_id] = float(family_repeat_cost)
                            feat_obj = build_candidate_features(
                                stage_name=str(stage_name),
                                candidate_label=str(pool[int(idx)].label),
                                candidate_family=str(family_id),
                                candidate_pool_index=int(idx),
                                position_id=int(pos),
                                append_position=int(append_position),
                                positions_considered=[int(x) for x in positions_considered_local],
                                gradient_signed=float(gradients[int(idx)]),
                                metric_proxy=float(candidate_metric_proxy),
                                sigma_hat=float(candidate_sigma_hat),
                                refit_window_indices=[int(i) for i in active_window_guess],
                                compile_cost=compile_est,
                                measurement_stats=meas_stats,
                                leakage_penalty=0.0,
                                stage_gate_open=bool(stage_gate_open),
                                leakage_gate_open=True,
                                trough_probe_triggered=bool(trough_probe_triggered_local),
                                trough_detected=False,
                                cfg=phase1_score_cfg,
                                cheap_score_cfg=(phase2_score_cfg if phase3_enabled else None),
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
                                family_repeat_cost=float(family_repeat_cost),
                            )
                            window_terms, window_labels = _window_terms_for_position(
                                selected_ops=list(selected_ops),
                                refit_window_indices=[int(i) for i in active_window_guess],
                                position_id=int(pos),
                            )
                            feat = dict(feat_obj.__dict__)
                            score_val = float(
                                feat.get("cheap_score", feat.get("simple_score", float("-inf")))
                            )
                            records_local.append(
                                {
                                    "feature": feat_obj,
                                    "cheap_score": float(score_val),
                                    "simple_score": float(feat.get("simple_score", float("-inf"))),
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
                positions_considered = [int(append_position)]
                score_eval = append_eval
                if backend_compile_oracle is not None:
                    finite_phase1_records = [
                        rec for rec in score_eval.get("records", [])
                        if math.isfinite(float(rec.get("cheap_score", rec.get("simple_score", float("-inf")))))
                    ]
                    if not finite_phase1_records:
                        stop_reason = "backend_compile_exhausted"
                        break
                trough = False
                phase1_last_probe_reason = "append_only"
                phase1_last_positions_considered = [int(x) for x in positions_considered]
                phase1_last_trough_detected = False
                phase1_last_trough_probe_triggered = False
                phase1_last_selected_score = float(score_eval["best_score"])
                best_feat = score_eval["best_feat"]
                best_idx = int(score_eval["best_idx"])
                selected_position = int(score_eval["best_position"])
                selection_mode = _cheap_selection_mode(best_feat, probe=False)
                controller_pre_snapshot = phase1_stage.pre_step_snapshot(
                    depth_local=int(depth),
                    max_depth=int(max_depth),
                )
                phase1_shortlist_score_key = _phase1_shortlist_score_key()
                controller_snapshot = phase1_stage.finalize_step_snapshot(
                    pre_snapshot=controller_pre_snapshot,
                    phase1_raw_scores=[
                        float(
                            rec.get(
                                phase1_shortlist_score_key,
                                rec.get("simple_score", float("-inf")),
                            )
                        )
                        for rec in score_eval.get("records", [])
                    ],
                )
                phase1_records = _attach_controller_snapshot(
                    list(score_eval.get("records", [])),
                    snapshot=controller_snapshot,
                )
                phase1_shortlisted_records = _phase_shortlist_with_legacy_hook(
                    phase1_records,
                    score_key=phase1_shortlist_score_key,
                    threshold=_controller_threshold(controller_snapshot, "phase1"),
                    cap=_controller_cap(controller_snapshot, "phase1", phase1_shortlist_size_val),
                    frontier_ratio=float(phase2_score_cfg.phase2_frontier_ratio),
                    tie_break_score_key="simple_score",
                    shortlist_flag="phase1_shortlisted",
                )
                phase1_last_retained_records = _candidate_feature_rows(phase1_shortlisted_records)
                if phase2_enabled:
                    cheap_records = [
                        {
                            **dict(rec),
                            "feature": rec.get("feature"),
                            "cheap_score": float(
                                rec.get("cheap_score", rec.get("simple_score", float("-inf")))
                            ),
                            "simple_score": float(rec.get("simple_score", float("-inf"))),
                            "candidate_pool_index": int(rec.get("candidate_pool_index", -1)),
                            "position_id": int(rec.get("position_id", append_position)),
                        }
                        for rec in phase1_shortlisted_records
                    ]
                    full_records: list[dict[str, Any]] = []
                    phase2_scaffold_context_cache: dict[tuple[int, ...], Any] = {}
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
                            runtime_split_chosen_representation_value: str = "parent",
                            runtime_split_child_indices_value: Sequence[int] | None = None,
                            runtime_split_child_labels_value: Sequence[str] | None = None,
                            runtime_split_child_generator_ids_value: Sequence[str] | None = None,
                        ) -> dict[str, Any]:
                            compiled_candidate = phase2_compiled_term_cache.get(str(candidate_label))
                            if compiled_candidate is None:
                                compiled_candidate = _compile_polynomial_action(
                                    candidate_term.polynomial,
                                    pauli_action_cache=pauli_action_cache,
                                )
                                phase2_compiled_term_cache[str(candidate_label)] = compiled_candidate
                            apsi_candidate = _apply_compiled_polynomial(
                                np.asarray(psi_current, dtype=complex),
                                compiled_candidate,
                            )
                            grad_candidate = float(
                                adapt_commutator_grad_from_hpsi(
                                    hpsi_current,
                                    apsi_candidate,
                                )
                            )
                            candidate_metric_proxy = _base_metric_for_candidate(
                                candidate_label=str(candidate_label),
                                candidate_term=candidate_term,
                                gradient_signed=float(grad_candidate),
                                precompiled_action=compiled_candidate,
                            )
                            candidate_sigma_hat = _phase3_sigma_hat_for_label(
                                candidate_label=str(candidate_label),
                                sigma_by_label=phase3_sigma_by_label,
                                phase3_enabled=phase3_enabled,
                            )
                            proxy_compile_est_candidate = phase1_compile_oracle.estimate(
                                candidate_term_count=int(len(compiled_candidate.terms)),
                                position_id=int(feat_base.position_id),
                                append_position=int(feat_base.append_position),
                                refit_active_count=int(len(feat_base.refit_window_indices)),
                                candidate_term=candidate_term,
                            )
                            compile_est_candidate = (
                                backend_compile_oracle.estimate_insertion(
                                    backend_compile_snapshot,
                                    candidate_term=candidate_term,
                                    position_id=int(feat_base.position_id),
                                    proxy_baseline=proxy_compile_est_candidate,
                                )
                                if backend_compile_oracle is not None and backend_compile_snapshot is not None
                                else proxy_compile_est_candidate
                            )
                            measurement_stats_candidate = phase1_measure_cache.estimate(
                                measurement_group_keys_for_term(candidate_term)
                            )
                            feat_candidate_base = build_candidate_features(
                                stage_name=str(feat_base.stage_name),
                                candidate_label=str(candidate_label),
                                candidate_family=str(feat_base.candidate_family),
                                candidate_pool_index=int(feat_base.candidate_pool_index),
                                position_id=int(feat_base.position_id),
                                append_position=int(feat_base.append_position),
                                positions_considered=[int(x) for x in feat_base.positions_considered],
                                gradient_signed=float(grad_candidate),
                                metric_proxy=float(candidate_metric_proxy),
                                sigma_hat=float(candidate_sigma_hat),
                                refit_window_indices=[int(i) for i in feat_base.refit_window_indices],
                                compile_cost=compile_est_candidate,
                                measurement_stats=measurement_stats_candidate,
                                leakage_penalty=0.0,
                                stage_gate_open=bool(feat_base.stage_gate_open),
                                leakage_gate_open=True,
                                trough_probe_triggered=bool(feat_base.trough_probe_triggered),
                                trough_detected=bool(feat_base.trough_detected),
                                cfg=phase1_score_cfg,
                                cheap_score_cfg=(phase2_score_cfg if phase3_enabled else None),
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
                                family_repeat_cost=float(feat_base.family_repeat_cost),
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
                                        "runtime_split_chosen_representation": str(
                                            runtime_split_chosen_representation_value
                                        ),
                                        "runtime_split_child_indices": (
                                            [int(x) for x in runtime_split_child_indices_value]
                                            if runtime_split_child_indices_value is not None
                                            else []
                                        ),
                                        "runtime_split_child_labels": (
                                            [str(x) for x in runtime_split_child_labels_value]
                                            if runtime_split_child_labels_value is not None
                                            else []
                                        ),
                                        "runtime_split_child_generator_ids": (
                                            [str(x) for x in runtime_split_child_generator_ids_value]
                                            if runtime_split_child_generator_ids_value is not None
                                            else []
                                        ),
                                    }
                                )
                            active_memory = phase2_memory_adapter.select_active(
                                phase2_optimizer_memory,
                                active_indices=list(feat_candidate_base.refit_window_indices),
                                source=f"adapt.depth{int(depth + 1)}.window_subset",
                            )
                            scaffold_key = tuple(int(i) for i in feat_candidate_base.refit_window_indices)
                            scaffold_context = phase2_scaffold_context_cache.get(scaffold_key)
                            if scaffold_context is None:
                                scaffold_context = phase2_novelty_oracle.prepare_scaffold_context(
                                    selected_ops=list(selected_ops),
                                    theta=np.asarray(theta_logical_current, dtype=float),
                                    psi_ref=np.asarray(psi_ref, dtype=complex),
                                    psi_state=np.asarray(psi_current, dtype=complex),
                                    h_compiled=h_compiled,
                                    hpsi_state=np.asarray(hpsi_current, dtype=complex),
                                    refit_window_indices=list(feat_candidate_base.refit_window_indices),
                                    pauli_action_cache=pauli_action_cache,
                                )
                                phase2_scaffold_context_cache[scaffold_key] = scaffold_context
                            feat_full = build_full_candidate_features(
                                base_feature=feat_candidate_base,
                                candidate_term=candidate_term,
                                cfg=phase2_score_cfg,
                                novelty_oracle=phase2_novelty_oracle,
                                curvature_oracle=phase2_curvature_oracle,
                                scaffold_context=scaffold_context,
                                h_compiled=h_compiled,
                                compiled_cache=phase2_compiled_term_cache,
                                pauli_action_cache=pauli_action_cache,
                                optimizer_memory=active_memory,
                                motif_library=(phase3_input_motif_library if phase3_enabled else None),
                                target_num_sites=int(num_sites),
                            )
                            return {
                                **dict(rec),
                                "feature": feat_full,
                                "cheap_score": float(
                                    feat_full.cheap_score
                                    if feat_full.cheap_score is not None
                                    else feat_full.simple_score or float("-inf")
                                ),
                                "simple_score": float(feat_full.simple_score or float("-inf")),
                                "phase2_raw_score": float(feat_full.phase2_raw_score or float("-inf")),
                                "full_v2_score": float(feat_full.full_v2_score or float("-inf")),
                                "candidate_pool_index": int(feat_full.candidate_pool_index),
                                "position_id": int(feat_full.position_id),
                                "candidate_term": candidate_term,
                            }

                        parent_record = _full_record_for_candidate(
                            candidate_term=rec["candidate_term"],
                            candidate_label=parent_label,
                            generator_metadata=parent_generator_meta,
                            symmetry_spec_candidate=parent_symmetry_spec,
                        )
                        candidate_variants = [parent_record]
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
                            split_child_records: list[dict[str, Any]] = []
                            split_child_record_by_generator_id: dict[str, dict[str, Any]] = {}
                            split_child_scores: dict[str, float] = {}
                            for child in split_children:
                                child_label = str(child.get("child_label"))
                                child_poly = child.get("child_polynomial")
                                child_meta = child.get("child_generator_metadata")
                                child_symmetry_gate = (
                                    dict(child.get("symmetry_gate", {}))
                                    if isinstance(child.get("symmetry_gate"), Mapping)
                                    else {}
                                )
                                if not isinstance(child_poly, PauliPolynomial):
                                    continue
                                if not isinstance(child_meta, Mapping):
                                    continue
                                pool_generator_registry[str(child_label)] = dict(child_meta)
                                phase3_runtime_split_summary["evaluated_child_count"] = int(
                                    phase3_runtime_split_summary.get("evaluated_child_count", 0)
                                ) + 1
                                if not bool(child_symmetry_gate.get("passed", True)):
                                    phase3_runtime_split_summary["rejected_child_count_symmetry"] = int(
                                        phase3_runtime_split_summary.get("rejected_child_count_symmetry", 0)
                                    ) + 1
                                child_record = _full_record_for_candidate(
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
                                    runtime_split_chosen_representation_value="child_atom",
                                    runtime_split_child_indices_value=(
                                        [int(child.get("child_index"))]
                                        if child.get("child_index") is not None
                                        else []
                                    ),
                                    runtime_split_child_labels_value=[str(child_label)],
                                    runtime_split_child_generator_ids_value=(
                                        [str(child_meta.get("generator_id"))]
                                        if child_meta.get("generator_id") is not None
                                        else []
                                    ),
                                )
                                split_child_records.append(dict(child_record))
                                split_child_scores[str(child_label)] = float(
                                    child_record.get("full_v2_score", float("-inf"))
                                )
                                if child_meta.get("generator_id") is not None:
                                    split_child_record_by_generator_id[str(child_meta.get("generator_id"))] = dict(child_record)
                            split_candidate_record: dict[str, Any] | None = None
                            admissible_child_subsets: list[list[str]] = []
                            best_split_choice_reason = "no_admissible_child_set"
                            best_split_gate_results: dict[str, Any] = {}
                            best_split_child_ids: list[str] = []
                            child_set_candidates = build_runtime_split_child_sets(
                                parent_label=str(parent_label),
                                family_id=str(feat_base.candidate_family),
                                num_sites=int(num_sites),
                                ordering=str(ordering),
                                qpb=int(max(1, qpb)),
                                split_mode=str(phase3_runtime_split_mode_key),
                                children=split_children,
                                parent_generator_metadata=parent_generator_meta,
                                symmetry_spec=parent_symmetry_spec,
                                max_subset_size=3,
                            )
                            phase3_runtime_split_summary["admissible_child_set_count"] = int(
                                phase3_runtime_split_summary.get("admissible_child_set_count", 0)
                            ) + int(len(child_set_candidates))
                            if child_set_candidates and split_child_records:
                                compat_oracle_split = CompatibilityPenaltyOracle(
                                    cfg=phase2_score_cfg,
                                    psi_state=np.asarray(psi_current, dtype=complex),
                                    compiled_cache=phase2_compiled_term_cache,
                                    pauli_action_cache=pauli_action_cache,
                                )
                                best_split_proxy = float("-inf")
                                best_split_payload: dict[str, Any] | None = None
                                for child_set in child_set_candidates:
                                    child_set_ids = [str(x) for x in child_set.get("child_generator_ids", [])]
                                    child_set_records = [
                                        dict(split_child_record_by_generator_id[str(child_id)])
                                        for child_id in child_set_ids
                                        if str(child_id) in split_child_record_by_generator_id
                                    ]
                                    if len(child_set_records) != len(child_set_ids):
                                        continue
                                    admissible_child_subsets.append(
                                        [str(x) for x in child_set.get("child_labels", [])]
                                    )
                                    penalty_total = 0.0
                                    for left_idx in range(len(child_set_records)):
                                        for right_idx in range(left_idx + 1, len(child_set_records)):
                                            penalty_total += float(
                                                compat_oracle_split.penalty(
                                                    child_set_records[left_idx],
                                                    child_set_records[right_idx],
                                                ).get("total", 0.0)
                                            )
                                    proxy_score = float(
                                        sum(
                                            float(rec_child.get("full_v2_score", float("-inf")))
                                            for rec_child in child_set_records
                                        )
                                        - penalty_total
                                    )
                                    if proxy_score > best_split_proxy:
                                        best_split_proxy = float(proxy_score)
                                        best_split_payload = dict(child_set)
                                if best_split_payload is not None:
                                    split_label = str(best_split_payload.get("candidate_label"))
                                    split_poly = best_split_payload.get("candidate_polynomial")
                                    split_meta = best_split_payload.get("candidate_generator_metadata")
                                    if isinstance(split_poly, PauliPolynomial) and isinstance(split_meta, Mapping):
                                        pool_generator_registry[str(split_label)] = dict(split_meta)
                                        best_split_gate_results = (
                                            dict(best_split_payload.get("symmetry_gate", {}))
                                            if isinstance(best_split_payload.get("symmetry_gate"), Mapping)
                                            else {}
                                        )
                                        best_split_child_ids = [
                                            str(x) for x in best_split_payload.get("child_generator_ids", [])
                                        ]
                                        split_candidate_record = _full_record_for_candidate(
                                            candidate_term=AnsatzTerm(
                                                label=str(split_label),
                                                polynomial=split_poly,
                                            ),
                                            candidate_label=str(split_label),
                                            generator_metadata=dict(split_meta),
                                            symmetry_spec_candidate=(
                                                dict(split_meta.get("symmetry_spec", {}))
                                                if isinstance(split_meta.get("symmetry_spec"), Mapping)
                                                else parent_symmetry_spec
                                            ),
                                            runtime_split_mode_value=str(phase3_runtime_split_mode_key),
                                            runtime_split_parent_label_value=str(parent_label),
                                            runtime_split_child_count_value=int(len(split_children)),
                                            runtime_split_chosen_representation_value="child_set",
                                            runtime_split_child_indices_value=[
                                                int(x) for x in best_split_payload.get("child_indices", [])
                                            ],
                                            runtime_split_child_labels_value=[
                                                str(x) for x in best_split_payload.get("child_labels", [])
                                            ],
                                            runtime_split_child_generator_ids_value=list(best_split_child_ids),
                                        )
                                        candidate_variants.append(dict(split_candidate_record))
                                        parent_score = float(parent_record.get("full_v2_score", float("-inf")))
                                        split_score = float(split_candidate_record.get("full_v2_score", float("-inf")))
                                        split_wins = bool(split_score > parent_score)
                                        if split_wins:
                                            phase3_runtime_split_summary["probe_child_set_count"] = int(
                                                phase3_runtime_split_summary.get("probe_child_set_count", 0)
                                            ) + 1
                                            best_split_choice_reason = "child_set_actual_score_better"
                                        else:
                                            phase3_runtime_split_summary["probe_parent_win_count"] = int(
                                                phase3_runtime_split_summary.get("probe_parent_win_count", 0)
                                            ) + 1
                                            best_split_choice_reason = "parent_actual_score_better"
                                        phase3_split_events.append(
                                            build_split_event(
                                                parent_generator_id=str(parent_generator_meta.get("generator_id")),
                                                child_generator_ids=list(best_split_child_ids),
                                                reason=f"depth{int(depth + 1)}_shortlist_probe",
                                                split_mode=str(phase3_runtime_split_mode_key),
                                                probe_trigger="phase2_shortlist",
                                                choice_reason=str(best_split_choice_reason),
                                                parent_score=float(parent_score),
                                                child_scores=dict(split_child_scores),
                                                admissible_child_subsets=list(admissible_child_subsets),
                                                chosen_representation=("child_set" if split_wins else "parent"),
                                                chosen_child_ids=(list(best_split_child_ids) if split_wins else []),
                                                split_margin=float(split_score - parent_score),
                                                symmetry_gate_results=dict(best_split_gate_results),
                                                compiled_cost_parent=float(
                                                    parent_record.get("feature").compile_cost_total
                                                )
                                                if isinstance(parent_record.get("feature"), CandidateFeatures)
                                                else None,
                                                compiled_cost_children=float(
                                                    split_candidate_record.get("feature").compile_cost_total
                                                )
                                                if isinstance(split_candidate_record.get("feature"), CandidateFeatures)
                                                else None,
                                                insertion_positions=[int(feat_base.position_id)],
                                            )
                                        )
                            elif split_children:
                                phase3_runtime_split_summary["probe_parent_win_count"] = int(
                                    phase3_runtime_split_summary.get("probe_parent_win_count", 0)
                                ) + 1
                                phase3_split_events.append(
                                    build_split_event(
                                        parent_generator_id=str(parent_generator_meta.get("generator_id")),
                                        child_generator_ids=[
                                            str(meta.get("generator_id"))
                                            for meta in (
                                                child.get("child_generator_metadata")
                                                for child in split_children
                                                if isinstance(child.get("child_generator_metadata"), Mapping)
                                            )
                                            if meta.get("generator_id") is not None
                                        ],
                                        reason=f"depth{int(depth + 1)}_shortlist_probe",
                                        split_mode=str(phase3_runtime_split_mode_key),
                                        probe_trigger="phase2_shortlist",
                                        choice_reason="no_admissible_child_set",
                                        parent_score=float(parent_record.get("full_v2_score", float("-inf"))),
                                        child_scores=dict(split_child_scores),
                                        admissible_child_subsets=[],
                                        chosen_representation="parent",
                                        chosen_child_ids=[],
                                        symmetry_gate_results={"admissible_child_set_count": 0},
                                        compiled_cost_parent=float(
                                            parent_record.get("feature").compile_cost_total
                                        )
                                        if isinstance(parent_record.get("feature"), CandidateFeatures)
                                        else None,
                                        insertion_positions=[int(feat_base.position_id)],
                                    )
                                )
                        candidate_variants = sorted(candidate_variants, key=_phase2_record_sort_key)
                        if candidate_variants:
                            full_records.append(dict(candidate_variants[0]))
                    full_records = sorted(full_records, key=_phase2_record_sort_key)
                    if backend_compile_oracle is not None:
                        finite_full_records = [
                            rec for rec in full_records
                            if math.isfinite(float(rec.get("full_v2_score", float("-inf"))))
                        ]
                        if not finite_full_records:
                            stop_reason = "backend_compile_exhausted"
                            break
                    full_records = _attach_controller_snapshot(
                        full_records,
                        snapshot=controller_snapshot,
                    )
                    phase2_last_shortlist_eval_records = [dict(rec) for rec in full_records]
                    phase2_shortlisted_records = _phase_shortlist_with_legacy_hook(
                        full_records,
                        score_key="phase2_raw_score",
                        threshold=_controller_threshold(controller_snapshot, "phase2"),
                        cap=_controller_cap(controller_snapshot, "phase2", phase2_score_cfg.shortlist_size),
                        frontier_ratio=float(phase2_score_cfg.phase2_frontier_ratio),
                        tie_break_score_key="cheap_score",
                        shortlist_flag="phase2_shortlisted",
                    )
                    phase2_last_geometric_shortlist_records = _candidate_feature_rows(
                        phase2_shortlisted_records
                    )
                    phase3_shortlisted_records = (
                        _phase_shortlist_with_legacy_hook(
                            phase2_shortlisted_records,
                            score_key="full_v2_score",
                            threshold=_controller_threshold(controller_snapshot, "phase3"),
                            cap=_controller_cap(controller_snapshot, "phase3", phase2_score_cfg.shortlist_size),
                            frontier_ratio=float(phase2_score_cfg.phase3_frontier_ratio),
                            tie_break_score_key="phase2_raw_score",
                            shortlist_flag="phase3_shortlisted",
                        )
                        if phase3_enabled
                        else list(phase2_shortlisted_records)
                    )
                    phase2_last_shortlist_records = _candidate_feature_rows(full_records)
                    phase2_last_retained_shortlist_records = _candidate_feature_rows(
                        phase3_shortlisted_records if phase3_enabled else phase2_shortlisted_records
                    )
                    if full_records:
                        if bool(phase2_enable_batching) and str(stage_name) == "core":
                            if phase3_enabled:
                                batch_source_records = (
                                    phase3_shortlisted_records
                                    if phase3_shortlisted_records
                                    else [dict(full_records[0])]
                                )
                                phase2_selected_records, batch_summary = reduced_plane_batch_select(
                                    batch_source_records,
                                    cfg=phase2_score_cfg,
                                    selected_ops=list(selected_ops),
                                    theta=np.asarray(theta_logical_current, dtype=float),
                                    psi_ref=np.asarray(psi_ref, dtype=complex),
                                    psi_state=np.asarray(psi_current, dtype=complex),
                                    h_compiled=h_compiled,
                                    novelty_oracle=phase2_novelty_oracle,
                                    curvature_oracle=phase2_curvature_oracle,
                                    compiled_cache=phase2_compiled_term_cache,
                                    pauli_action_cache=pauli_action_cache,
                                    tie_break_score_key="phase2_raw_score",
                                )
                                phase2_last_batch_penalty_total = float(
                                    batch_summary.get("additivity_defect", 0.0)
                                )
                                if not phase2_selected_records:
                                    phase2_selected_records = [dict(full_records[0])]
                            else:
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
                                    tie_break_score_key="cheap_score",
                                )
                        else:
                            phase2_selected_records = [dict(full_records[0])]
                        phase2_selected_records = sorted(phase2_selected_records, key=_phase2_record_sort_key)
                        phase2_last_batch_selected = bool(len(phase2_selected_records) > 1)
                        phase2_last_admitted_records = _candidate_feature_rows(phase2_selected_records)
                        top_feat = phase2_selected_records[0].get("feature")
                        if isinstance(top_feat, CandidateFeatures):
                            phase1_feature_selected = dict(top_feat.__dict__)
                            phase1_feature_selected["trough_detected"] = bool(trough)
                            phase1_last_selected_score = float(
                                top_feat.selector_score
                                if top_feat.selector_score is not None
                                else (
                                    top_feat.full_v2_score
                                    if top_feat.full_v2_score is not None
                                    else (
                                        top_feat.phase2_raw_score
                                        if top_feat.phase2_raw_score is not None
                                        else top_feat.simple_score or float("-inf")
                                    )
                                )
                            )
                            best_idx = int(top_feat.candidate_pool_index)
                            selected_position = int(top_feat.position_id)
                            split_selected = bool(str(top_feat.runtime_split_mode) != "off")
                            selection_mode = (
                                "phase3_rerank_split"
                                if phase3_enabled and split_selected
                                else ("phase3_rerank" if phase3_enabled else "phase2_raw")
                            )
                    elif best_feat is not None:
                        best_feat["trough_detected"] = bool(trough)
                        phase1_feature_selected = dict(best_feat)
                        phase1_last_selected_score = float(best_feat.get("simple_score", float("-inf")))
                        best_idx = int(best_feat.get("candidate_pool_index", best_idx))
                        selected_position = int(best_feat.get("position_id", selected_position))
                        selection_mode = "simple_v1_full_zero_fallback"
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
                if seq2p_logical_mode:
                    if allow_repeats:
                        repeat_bias = 1.5
                        logical_scores = logical_grad_scores / (1.0 + repeat_bias * logical_selection_counts.astype(float))
                        best_logical_idx = int(
                            max(logical_available_indices, key=lambda idx: float(logical_scores[int(idx)]))
                        )
                    else:
                        best_logical_idx = int(
                            max(logical_available_indices, key=lambda idx: float(logical_grad_scores[int(idx)]))
                        )
                    logical_candidate = logical_candidates[int(best_logical_idx)]
                    best_idx = int(logical_candidate.pool_indices[0])
                    selected_logical_label = str(logical_candidate.logical_label)
                    selected_logical_size = int(len(logical_candidate.pool_indices))
                    selected_logical_pool_indices = [int(x) for x in logical_candidate.pool_indices]
                    selected_grad_signed_components = list(
                        logical_grad_signed_components_all[int(best_logical_idx)]
                    )
                    selected_grad_abs_components = list(
                        logical_grad_abs_components_all[int(best_logical_idx)]
                    )
                    selection_mode = "gradient_seq2p"
                else:
                    if allow_repeats:
                        repeat_bias = 1.5
                        scores = grad_magnitudes / (1.0 + repeat_bias * selection_counts.astype(float))
                        best_idx = int(np.argmax(scores))
                    else:
                        best_idx = int(np.argmax(grad_magnitudes))
                    selection_mode = "gradient"

            if candidate_gradient_scout:
                for scout_row in candidate_gradient_scout:
                    scout_row["selected_for_optimization"] = (
                        int(scout_row.get("candidate_pool_index", -1)) == int(best_idx)
                    )
                phase3_last_candidate_gradient_scout = [dict(row) for row in candidate_gradient_scout]

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
                        else (
                            str(selected_logical_label)
                            if selected_logical_label is not None
                            else str(pool[best_idx].label)
                        )
                    )
                ),
                selected_position=int(selected_position),
                stage_name=str(stage_name),
                selection_score=(float(phase1_last_selected_score) if phase1_enabled else None),
                energy=float(energy_current),
            )

            # 3) Check gradient convergence (with optional finite-angle fallback)
            if not phase1_enabled and not seq2p_logical_mode:
                selection_mode = "gradient"
            init_theta = 0.0
            init_theta_values: list[float] = (
                [0.0] * int(selected_logical_size)
                if seq2p_logical_mode and selected_logical_size > 1
                else [0.0]
            )
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
                            trial_ops, trial_theta_vec = _splice_candidate_at_position(
                                ops=selected_ops,
                                theta=np.asarray(theta, dtype=float),
                                op=pool[int(idx)],
                                position_id=int(append_position),
                                init_theta=float(trial_theta),
                            )
                            if adapt_state_backend_key == "compiled":
                                trial_executor = fallback_executor_cache.get(int(idx))
                                if trial_executor is None:
                                    trial_executor = _build_compiled_executor(trial_ops)
                                    fallback_executor_cache[int(idx)] = trial_executor
                                psi_trial = _prepare_selected_state(
                                    ops_now=trial_ops,
                                    theta_now=trial_theta_vec,
                                    executor_now=trial_executor,
                                    parameter_layout_now=_build_selected_layout(trial_ops),
                                )
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
                    fallback_best_probe_theta = (
                        float(best_probe_theta) if best_probe_theta is not None else None
                    )
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
                            selected_position=int(selected_position),
                            init_theta=float(init_theta),
                            probe_delta_e=float(fallback_best_probe_delta_e),
                        )
                    else:
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
                                    else (
                                        feat_rescue.cheap_score
                                        if feat_rescue.cheap_score is not None
                                        else feat_rescue.simple_score or float("-inf")
                                    )
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
                    if bool(eps_grad_termination_enabled):
                        stop_reason = "eps_grad"
                        _ai_log(
                            "hardcoded_adapt_converged_grad",
                            max_grad=float(max_grad),
                            eps_grad=float(eps_grad),
                        )
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

            if _selector_debug_enabled_for_depth(int(depth + 1)):
                _ai_log(
                    "hardcoded_adapt_phase3_selector_debug",
                    **_selector_debug_payload(
                        depth_one_based=int(depth + 1),
                        beam_enabled=False,
                        selection_mode_value=str(selection_mode),
                        stage_name_value=str(stage_name),
                        selected_feature_row=phase1_feature_selected,
                        scored_rows=phase2_last_shortlist_eval_records,
                        phase2_rows=phase2_shortlisted_records,
                        phase3_rows=phase3_shortlisted_records,
                        admitted_rows=phase2_selected_records,
                        split_summary=phase3_runtime_split_summary,
                    ),
                )

            # 4) Admit selected operator (append or insertion in continuation modes).
            selected_batch_records_for_history: list[dict[str, Any]] = []
            selected_batch_labels: list[str] = []
            selected_batch_positions: list[int] = []
            selected_batch_indices: list[int] = []
            selected_batch_measurement_keys: list[str] = []
            selected_new_parameter_indices: list[int] = []
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
                    admitted_layout = _build_selected_layout([admitted_term])
                    runtime_insert_pos = int(runtime_insert_position(selected_layout, int(pos_eff)))
                    if phase2_enabled:
                        phase2_optimizer_memory = phase2_memory_adapter.remap_insert(
                            phase2_optimizer_memory,
                            position_id=int(runtime_insert_pos),
                            count=int(admitted_layout.runtime_parameter_count),
                        )
                    selected_ops, theta = _splice_candidate_at_position(
                        ops=selected_ops,
                        theta=np.asarray(theta, dtype=float),
                        op=admitted_term,
                        position_id=int(pos_eff),
                        init_theta=0.0,
                    )
                    selected_layout = _build_selected_layout(selected_ops)
                    if (
                        phase3_enabled
                        and str(feat_rec.runtime_split_mode) != "off"
                        and feat_rec.parent_generator_id is not None
                    ):
                        selected_child_generator_ids = [str(x) for x in feat_rec.runtime_split_child_generator_ids]
                        phase3_split_events.append(
                            build_split_event(
                                parent_generator_id=str(feat_rec.parent_generator_id),
                                child_generator_ids=(
                                    list(selected_child_generator_ids)
                                    if selected_child_generator_ids
                                    else ([str(feat_rec.generator_id)] if feat_rec.generator_id is not None else [])
                                ),
                                reason=f"depth{int(depth + 1)}_selected",
                                split_mode=str(feat_rec.runtime_split_mode),
                                choice_reason="selected_for_admission",
                                chosen_representation=str(feat_rec.runtime_split_chosen_representation),
                                chosen_child_ids=list(selected_child_generator_ids),
                                symmetry_gate_results=(
                                    dict(feat_rec.generator_metadata.get("compile_metadata", {}).get("runtime_split", {}).get("symmetry_gate", {}))
                                    if isinstance(feat_rec.generator_metadata, Mapping)
                                    and isinstance(feat_rec.generator_metadata.get("compile_metadata"), Mapping)
                                    and isinstance(
                                        feat_rec.generator_metadata.get("compile_metadata", {}).get("runtime_split"),
                                        Mapping,
                                    )
                                    else {}
                                ),
                                insertion_positions=[int(pos_eff)],
                            )
                        )
                        if str(feat_rec.runtime_split_chosen_representation) == "child_set":
                            phase3_runtime_split_summary["selected_child_set_count"] = int(
                                phase3_runtime_split_summary.get("selected_child_set_count", 0)
                            ) + 1
                        phase3_runtime_split_summary["selected_child_count"] = int(
                            phase3_runtime_split_summary.get("selected_child_count", 0)
                        ) + int(
                            max(
                                1,
                                len(selected_child_generator_ids)
                                if selected_child_generator_ids
                                else len(feat_rec.runtime_split_child_labels),
                            )
                        )
                        phase3_runtime_split_summary["selected_child_labels"] = [
                            *list(phase3_runtime_split_summary.get("selected_child_labels", [])),
                            *(
                                [str(x) for x in feat_rec.runtime_split_child_labels]
                                if feat_rec.runtime_split_child_labels
                                else [str(admitted_term.label)]
                            ),
                        ]
                    original_positions_seen.append(int(pos_orig))
                    selection_counts[idx_sel] += 1
                    if not allow_repeats:
                        available_indices.discard(idx_sel)
                    selected_batch_records_for_history.append(dict(feat_rec.__dict__))
                    selected_batch_labels.append(str(admitted_term.label))
                    selected_batch_positions.append(int(pos_orig))
                    selected_batch_indices.append(int(idx_sel))
                    selected_batch_measurement_keys.extend(measurement_group_keys_for_term(admitted_term))
                if selected_batch_indices:
                    best_idx = int(selected_batch_indices[0])
                    selected_position = int(selected_batch_positions[0])
            elif phase1_enabled:
                admitted_term = pool[int(best_idx)]
                admitted_layout = _build_selected_layout([admitted_term])
                runtime_insert_pos = int(runtime_insert_position(selected_layout, int(selected_position)))
                if phase2_enabled:
                    phase2_optimizer_memory = phase2_memory_adapter.remap_insert(
                        phase2_optimizer_memory,
                        position_id=int(runtime_insert_pos),
                        count=int(admitted_layout.runtime_parameter_count),
                    )
                selected_ops, theta = _splice_candidate_at_position(
                    ops=selected_ops,
                    theta=np.asarray(theta, dtype=float),
                    op=admitted_term,
                    position_id=int(selected_position),
                    init_theta=float(init_theta),
                )
                selected_layout = _build_selected_layout(selected_ops)
                selection_counts[best_idx] += 1
                if not allow_repeats:
                    available_indices.discard(best_idx)
                if isinstance(phase1_feature_selected, dict):
                    selected_batch_records_for_history.append(dict(phase1_feature_selected))
                selected_batch_labels.append(str(pool[int(best_idx)].label))
                selected_batch_positions.append(int(selected_position))
                selected_batch_indices.append(int(best_idx))
                selected_batch_measurement_keys.extend(measurement_group_keys_for_term(pool[int(best_idx)]))
            else:
                selected_ops, theta = _splice_candidate_at_position(
                    ops=selected_ops,
                    theta=np.asarray(theta, dtype=float),
                    op=pool[int(best_idx)],
                    position_id=int(len(selected_ops)),
                    init_theta=float(init_theta),
                )
                selected_layout = _build_selected_layout(selected_ops)
                selection_counts[best_idx] += 1
                if not allow_repeats:
                    available_indices.discard(best_idx)
                selected_batch_labels.append(str(pool[int(best_idx)].label))
                selected_batch_positions.append(int(selected_position))
                selected_batch_indices.append(int(best_idx))
                selected_batch_measurement_keys.extend(measurement_group_keys_for_term(pool[int(best_idx)]))
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
                energy_obj_val = _evaluate_selected_energy_objective(
                    ops_now=selected_ops,
                    theta_now=np.asarray(x, dtype=float),
                    executor_now=selected_executor,
                    parameter_layout_now=selected_layout,
                    objective_stage="depth_opt",
                    depth_marker=int(depth + 1),
                )
                if adapt_inner_optimizer_key in {"COBYLA", "POWELL"}:
                    cobyla_nfev_so_far += 1
                    if energy_obj_val < cobyla_best_fun:
                        cobyla_best_fun = float(energy_obj_val)
                    now = time.perf_counter()
                    if (now - cobyla_last_hb_t) >= float(adapt_spsa_progress_every_s):
                        _ai_log(
                            "hardcoded_adapt_scipy_heartbeat",
                            stage="depth_opt",
                            depth=int(depth + 1),
                            opt_method=str(adapt_inner_optimizer_key),
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
            n_theta_runtime = int(theta.size)
            theta_logical_selected = _logical_theta_alias(theta, selected_layout)
            n_theta_logical = int(theta_logical_selected.size)
            depth_local = int(depth + 1)
            depth_cumulative = int(adapt_ref_base_depth) + int(depth_local)
            periodic_full_refit_triggered = bool(
                adapt_reopt_policy_key == "windowed"
                and adapt_full_refit_every_val > 0
                and depth_cumulative % adapt_full_refit_every_val == 0
            )
            reopt_active_indices, reopt_policy_effective = _resolve_reopt_active_indices(
                policy=adapt_reopt_policy_key,
                n=n_theta_logical,
                theta=theta_logical_selected,
                window_size=adapt_window_size_val,
                window_topk=adapt_window_topk_val,
                periodic_full_refit_triggered=periodic_full_refit_triggered,
            )
            reopt_runtime_active_indices = runtime_indices_for_logical_indices(
                selected_layout,
                reopt_active_indices,
            )
            inherited_reopt_indices = [
                int(i) for i in reopt_active_indices
                if (not phase1_enabled) or int(i) != int(selected_position)
            ]
            if phase1_enabled and isinstance(phase1_feature_selected, dict):
                phase1_feature_selected["refit_window_indices"] = [int(i) for i in inherited_reopt_indices]
            if phase1_enabled and selected_batch_records_for_history:
                for rec in selected_batch_records_for_history:
                    rec["refit_window_indices"] = [int(i) for i in inherited_reopt_indices]
            _obj_opt, opt_x0 = _make_reduced_objective(theta, reopt_runtime_active_indices, _obj)
            phase2_active_memory = None
            phase2_last_optimizer_memory_reused = False
            phase2_last_optimizer_memory_source = "unavailable"
            if phase2_enabled and adapt_inner_optimizer_key == "SPSA":
                phase2_active_memory = phase2_memory_adapter.select_active(
                    phase2_optimizer_memory,
                    active_indices=list(reopt_runtime_active_indices),
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
                if len(reopt_runtime_active_indices) == n_theta_runtime:
                    theta = np.asarray(result.x, dtype=float)
                else:
                    result_x = np.asarray(result.x, dtype=float).ravel()
                    for k, idx in enumerate(reopt_runtime_active_indices):
                        theta[idx] = float(result_x[k])
                energy_current = float(result.fun)
                nfev_opt = int(result.nfev)
                nit_opt = int(result.nit)
                opt_success = bool(result.success)
                opt_message = str(result.message)
                if phase2_enabled:
                    phase2_optimizer_memory = phase2_memory_adapter.merge_active(
                        phase2_optimizer_memory,
                        active_indices=list(reopt_runtime_active_indices),
                        active_state=phase2_memory_adapter.from_result(
                            result,
                            method=str(adapt_inner_optimizer_key),
                            parameter_count=int(len(reopt_runtime_active_indices)),
                            source=f"adapt.depth{int(depth + 1)}.spsa_result",
                        ),
                        source=f"adapt.depth{int(depth + 1)}.merge",
                    )
            else:
                if scipy_minimize is None:
                    raise RuntimeError(
                        f"SciPy minimize is unavailable for {adapt_inner_optimizer_key} ADAPT inner optimizer."
                    )
                result = scipy_minimize(
                    _obj_opt,
                    opt_x0,
                    method=str(adapt_inner_optimizer_key),
                    options=_scipy_inner_options(int(maxiter)),
                )
                # Reconstruct full theta from reduced optimizer result
                if len(reopt_runtime_active_indices) == n_theta_runtime:
                    theta = np.asarray(result.x, dtype=float)
                else:
                    result_x = np.asarray(result.x, dtype=float).ravel()
                    for k, idx in enumerate(reopt_runtime_active_indices):
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
                # HH staged policy is energy-drop first; grad floors remain diagnostic.
                if bool(drop_low_signal):
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
            selected_logical_primary_label = (
                str(selected_logical_label)
                if selected_logical_label is not None
                else str(selected_primary_label)
            )
            selected_grad_signed_value = (
                float(selected_grad_signed_components[0])
                if selected_logical_label is not None and selected_grad_signed_components
                else (
                    float(phase1_feature_selected.get("g_signed"))
                    if isinstance(phase1_feature_selected, dict)
                    and phase1_feature_selected.get("g_signed") is not None
                    else float(gradients[best_idx])
                )
            )
            selected_grad_abs_value = (
                float(selected_grad_abs_components[0])
                if selected_logical_label is not None and selected_grad_abs_components
                else (
                    float(phase1_feature_selected.get("g_abs"))
                    if isinstance(phase1_feature_selected, dict)
                    and phase1_feature_selected.get("g_abs") is not None
                    else float(grad_magnitudes[best_idx])
                )
            )
            selected_logical_grad_abs_value = (
                float(math.sqrt(sum(float(x) * float(x) for x in selected_grad_abs_components)))
                if selected_logical_label is not None and selected_grad_abs_components
                else float(selected_grad_abs_value)
            )
            history_row = {
                "depth": int(depth + 1),
                "selected_op": str(selected_primary_label),
                "selected_logical_op": str(selected_logical_primary_label),
                "selected_logical_size": int(selected_logical_size),
                "selected_logical_pool_indices": (
                    [int(x) for x in selected_logical_pool_indices]
                    if selected_logical_pool_indices
                    else [int(best_idx)]
                ),
                "pool_index": int(best_idx),
                "selected_ops": [str(x) for x in selected_batch_labels],
                "selected_pool_indices": [int(x) for x in selected_batch_indices],
                "selection_mode": str(selection_mode),
                "init_theta": (
                    float(init_theta_values[0])
                    if selected_logical_label is not None and init_theta_values
                    else float(init_theta)
                ),
                "init_theta_values": (
                    [float(x) for x in init_theta_values]
                    if selected_logical_label is not None
                    else [float(init_theta)]
                ),
                "max_grad": float(max_grad),
                "selected_grad_signed": float(selected_grad_signed_value),
                "selected_grad_abs": float(selected_grad_abs_value),
                "selected_logical_grad_abs": float(selected_logical_grad_abs_value),
                "selected_grad_signed_components": (
                    [float(x) for x in selected_grad_signed_components]
                    if selected_grad_signed_components
                    else [float(selected_grad_signed_value)]
                ),
                "selected_grad_abs_components": (
                    [float(x) for x in selected_grad_abs_components]
                    if selected_grad_abs_components
                    else [float(selected_grad_abs_value)]
                ),
                "parameterization": ("double_sequential" if selected_logical_label is not None else "single_term"),
                "fallback_scan_size": int(fallback_scan_size),
                "fallback_best_probe_delta_e": (
                    float(fallback_best_probe_delta_e) if fallback_best_probe_delta_e is not None else None
                ),
                "fallback_best_probe_theta": (
                    float(fallback_best_probe_theta[0])
                    if isinstance(fallback_best_probe_theta, (list, tuple, np.ndarray))
                    else (float(fallback_best_probe_theta) if fallback_best_probe_theta is not None else None)
                ),
                "fallback_best_probe_theta_values": (
                    [float(x) for x in fallback_best_probe_theta]
                    if isinstance(fallback_best_probe_theta, (list, tuple, np.ndarray))
                    else (
                        [float(fallback_best_probe_theta)]
                        if fallback_best_probe_theta is not None
                        else None
                    )
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
                "gradient_source": str(phase3_gradient_source_name),
                "max_gradient_stderr": float(phase3_last_max_gradient_stderr),
                "candidate_gradient_scout": [dict(row) for row in candidate_gradient_scout],
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
                "reopt_runtime_active_indices": [int(i) for i in reopt_runtime_active_indices],
                "reopt_runtime_active_count": int(len(reopt_runtime_active_indices)),
                "num_parameters_after_opt": int(theta.size),
                "logical_num_parameters_after_opt": int(len(selected_ops)),
                "parameters_added_this_step": int(theta.size - theta_before_opt.size),
                "logical_parameters_added_this_step": int(len(selected_batch_labels)) if selected_batch_labels else 1,
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
                        "selector_score": _selector_score_value(phase1_feature_selected),
                        "selector_burden": _selector_burden_value(phase1_feature_selected),
                        "selected_feature_rows": [dict(x) for x in selected_batch_records_for_history],
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
                        "cheap_score": (
                            float(phase1_feature_selected.get("cheap_score"))
                            if isinstance(phase1_feature_selected, dict)
                            and phase1_feature_selected.get("cheap_score") is not None
                            else None
                        ),
                        "cheap_score_version": (
                            str(phase1_feature_selected.get("cheap_score_version"))
                            if isinstance(phase1_feature_selected, dict)
                            and phase1_feature_selected.get("cheap_score_version") is not None
                            else None
                        ),
                        "cheap_metric_proxy": (
                            float(phase1_feature_selected.get("cheap_metric_proxy"))
                            if isinstance(phase1_feature_selected, dict)
                            and phase1_feature_selected.get("cheap_metric_proxy") is not None
                            else None
                        ),
                        "cheap_benefit_proxy": (
                            float(phase1_feature_selected.get("cheap_benefit_proxy"))
                            if isinstance(phase1_feature_selected, dict)
                            and phase1_feature_selected.get("cheap_benefit_proxy") is not None
                            else None
                        ),
                        "cheap_burden_total": (
                            float(phase1_feature_selected.get("cheap_burden_total"))
                            if isinstance(phase1_feature_selected, dict)
                            and phase1_feature_selected.get("cheap_burden_total") is not None
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
                            and phase1_feature_selected.get("g_lcb") is not None
                            else None
                        ),
                        "sigma_hat": (
                            float(phase1_feature_selected.get("sigma_hat"))
                            if isinstance(phase1_feature_selected, dict)
                            and phase1_feature_selected.get("sigma_hat") is not None
                            else None
                        ),
                        "F_raw": (
                            float(phase1_feature_selected.get("F_raw"))
                            if isinstance(phase1_feature_selected, dict)
                            and phase1_feature_selected.get("F_raw") is not None
                            else None
                        ),
                        "F_red": (
                            float(phase1_feature_selected.get("F_red"))
                            if isinstance(phase1_feature_selected, dict)
                            and phase1_feature_selected.get("F_red") is not None
                            else None
                        ),
                        "h_eff": (
                            float(phase1_feature_selected.get("h_eff"))
                            if isinstance(phase1_feature_selected, dict)
                            and phase1_feature_selected.get("h_eff") is not None
                            else None
                        ),
                        "ridge_used": (
                            float(phase1_feature_selected.get("ridge_used"))
                            if isinstance(phase1_feature_selected, dict)
                            and phase1_feature_selected.get("ridge_used") is not None
                            else None
                        ),
                        "family_repeat_cost": (
                            float(phase1_feature_selected.get("family_repeat_cost", 0.0))
                            if isinstance(phase1_feature_selected, dict)
                            else None
                        ),
                        "refit_window_indices": [int(i) for i in reopt_active_indices],
                        "compile_cost_proxy": (
                            dict(phase1_feature_selected.get("compiled_position_cost_proxy", {}))
                            if isinstance(phase1_feature_selected, dict)
                            else None
                        ),
                        "compile_cost_mode": str(phase3_backend_cost_mode_key),
                        "compile_cost_source": (
                            str(phase1_feature_selected.get("compile_cost_source", "proxy"))
                            if isinstance(phase1_feature_selected, dict)
                            else "proxy"
                        ),
                        "compile_cost_total": (
                            float(phase1_feature_selected.get("compile_cost_total", 0.0))
                            if isinstance(phase1_feature_selected, dict)
                            else 0.0
                        ),
                        "compile_gate_open": (
                            bool(phase1_feature_selected.get("compile_gate_open", True))
                            if isinstance(phase1_feature_selected, dict)
                            else True
                        ),
                        "compile_failure_reason": (
                            phase1_feature_selected.get("compile_failure_reason")
                            if isinstance(phase1_feature_selected, dict)
                            else None
                        ),
                        "compile_cost_backend": (
                            dict(phase1_feature_selected.get("compiled_position_cost_backend", {}))
                            if isinstance(phase1_feature_selected, dict)
                            and isinstance(phase1_feature_selected.get("compiled_position_cost_backend"), Mapping)
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
                            "phase2_raw_score": (
                                float(phase1_feature_selected.get("phase2_raw_score"))
                                if isinstance(phase1_feature_selected, dict)
                                and phase1_feature_selected.get("phase2_raw_score") is not None
                                else None
                            ),
                            "full_v2_score": (
                                float(phase1_feature_selected.get("full_v2_score"))
                                if isinstance(phase1_feature_selected, dict)
                                and phase1_feature_selected.get("full_v2_score") is not None
                                else None
                            ),
                            "shortlist_size": int(len(phase2_last_shortlist_records)),
                            "shortlisted_records": [dict(x) for x in phase2_last_shortlist_records],
                            "scored_surface_size": int(len(phase2_last_shortlist_records)),
                            "scored_surface_records": [dict(x) for x in phase2_last_shortlist_records],
                            "retained_shortlist_size": int(len(phase2_last_retained_shortlist_records)),
                            "retained_shortlist_records": [
                                dict(x) for x in phase2_last_retained_shortlist_records
                            ],
                            "admitted_record_count": int(len(phase2_last_admitted_records)),
                            "admitted_records": [dict(x) for x in phase2_last_admitted_records],
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
                            "runtime_split_chosen_representation": (
                                str(phase1_feature_selected.get("runtime_split_chosen_representation", "parent"))
                                if isinstance(phase1_feature_selected, dict)
                                else "parent"
                            ),
                            "runtime_split_child_indices": (
                                [int(x) for x in phase1_feature_selected.get("runtime_split_child_indices", [])]
                                if isinstance(phase1_feature_selected, dict)
                                else []
                            ),
                            "runtime_split_child_labels": (
                                [str(x) for x in phase1_feature_selected.get("runtime_split_child_labels", [])]
                                if isinstance(phase1_feature_selected, dict)
                                else []
                            ),
                            "runtime_split_child_generator_ids": (
                                [str(x) for x in phase1_feature_selected.get("runtime_split_child_generator_ids", [])]
                                if isinstance(phase1_feature_selected, dict)
                                else []
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
                phase1_stage.record_admission(
                    selector_step=int(depth + 1),
                    energy_before=float(energy_prev),
                    energy_after_refit=float(energy_current),
                )
                if selected_batch_records_for_history:
                    for rec in selected_batch_records_for_history:
                        phase1_features_history.append(dict(rec))
                elif isinstance(phase1_feature_selected, dict):
                    phase1_features_history.append(dict(phase1_feature_selected))
                phase1_measure_cache.commit(
                    selected_batch_measurement_keys
                    if selected_batch_measurement_keys
                    else measurement_group_keys_for_term(pool[int(best_idx)])
                )
                phase1_prune_metadata_state, phase1_prune_first_seen_steps = _transport_prune_metadata_after_admission(
                    metadata_rows=phase1_prune_metadata_state,
                    labels_added=[str(x) for x in selected_batch_labels],
                    positions_added=[int(x) for x in selected_batch_positions],
                    feature_rows_added=[
                        dict(x) if isinstance(x, Mapping) else {}
                        for x in selected_batch_records_for_history
                    ],
                    selector_step=int(depth + 1),
                    first_seen_steps=phase1_prune_first_seen_steps,
                )
                prune_controller_snapshot = None
                if selected_batch_records_for_history and isinstance(selected_batch_records_for_history[0], Mapping):
                    prune_controller_snapshot = selected_batch_records_for_history[0].get("controller_snapshot")
                elif isinstance(phase1_feature_selected, Mapping):
                    prune_controller_snapshot = phase1_feature_selected.get("controller_snapshot")
                selected_ops, theta, energy_current, phase2_optimizer_memory, phase1_prune_metadata_state, phase1_prune_first_seen_steps, prune_summary = _execute_live_mature_prune_pass(
                    ops_now=list(selected_ops),
                    theta_now=np.asarray(theta, dtype=float),
                    energy_now=float(energy_current),
                    optimizer_memory_now=dict(phase2_optimizer_memory),
                    metadata_rows=phase1_prune_metadata_state,
                    first_seen_steps=phase1_prune_first_seen_steps,
                    controller_snapshot=(prune_controller_snapshot if isinstance(prune_controller_snapshot, Mapping) else None),
                    selector_step=int(depth + 1),
                    admitted_gain=float(max(0.0, float(energy_prev) - float(energy_current))),
                    history_rows=history,
                )
                history[-1]["post_admission_prune"] = copy.deepcopy(prune_summary)
                selected_layout = _build_selected_layout(selected_ops)
                if adapt_state_backend_key == "compiled":
                    selected_executor = _build_compiled_executor(selected_ops) if len(selected_ops) > 0 else None
                else:
                    selected_executor = None

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
                    energy_obj_val = _evaluate_selected_energy_objective(
                        ops_now=selected_ops,
                        theta_now=np.asarray(x, dtype=float),
                        executor_now=selected_executor,
                        parameter_layout_now=selected_layout,
                        objective_stage="final_full_refit",
                        depth_marker=int(len(history)),
                    )
                    if adapt_inner_optimizer_key in {"COBYLA", "POWELL"}:
                        cobyla_nfev_so_far += 1
                        if energy_obj_val < cobyla_best_fun:
                            cobyla_best_fun = float(energy_obj_val)
                        now = time.perf_counter()
                        if (now - cobyla_last_hb_t) >= float(adapt_spsa_progress_every_s):
                            _ai_log(
                                "hardcoded_adapt_scipy_heartbeat",
                                stage="final_full_refit",
                                opt_method=str(adapt_inner_optimizer_key),
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
                        raise RuntimeError(
                            f"SciPy minimize is unavailable for {adapt_inner_optimizer_key} final full refit."
                        )
                    final_result = scipy_minimize(
                        _obj_final,
                        final_x0,
                        method=str(adapt_inner_optimizer_key),
                        options=_scipy_inner_options(int(maxiter)),
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

        prune_summary = (
            dict(prune_summary)
            if isinstance(prune_summary, Mapping)
            else _default_prune_summary(reason="final_checkpoint", energy=float(energy_current))
        )
        if (not phase1_prune_live_mode) and phase1_enabled and bool(phase1_prune_enabled) and int(len(selected_ops)) > 1:
            phase1_scaffold_pre_prune = {
                "operators": [str(op.label) for op in selected_ops],
                "optimal_point": [float(x) for x in np.asarray(theta, dtype=float).tolist()],
                "energy": float(energy_current),
            }
            prune_cfg = PruneConfig(
                max_candidates=int(max(1, phase1_prune_max_candidates)),
                min_candidates=1,
                fraction_candidates=float(max(0.0, phase1_prune_fraction)),
                max_regression=float(max(0.0, phase1_prune_max_regression)),
                retained_gain_ratio=0.5,
                protect_steps=2,
                stale_age=2,
                stagnation_threshold=0.0,
                small_theta_abs=1e-3,
                small_theta_relative=0.5,
                cooldown_steps=2,
                local_window_size=4,
                old_fraction=0.25,
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
                    benefit = row.get("selector_score", None)
                    burden = row.get("selector_burden", 0.0)
                    if benefit is None:
                        benefit = row.get("full_v2_score", None)
                    if benefit is None:
                        benefit = row.get("cheap_score", row.get("simple_score", None))
                    if benefit is None:
                        benefit = row.get("metric_proxy", row.get("selected_grad_abs", float("inf")))
                    benefit_f = float(benefit) / float(1.0 + max(0.0, float(burden or 0.0)))
                    if not math.isfinite(benefit_f):
                        benefit_f = float("inf")
                    benefits.insert(pos, benefit_f)
                while len(benefits) < int(len(selected_ops)):
                    benefits.append(float("inf"))
                return [float(x) for x in benefits[: int(len(selected_ops))]]

            def _fallback_prune_metadata(
                *,
                label: str,
                theta_value: float,
                index: int,
            ) -> ScaffoldCoordinateMetadata:
                theta_abs = abs(float(theta_value))
                return ScaffoldCoordinateMetadata(
                    candidate_label=str(label),
                    generator_id=None,
                    admission_step=0,
                    first_seen_step=0,
                    selector_score=0.0,
                    selector_burden=0.0,
                    cooldown_remaining=0,
                    cumulative_abs_motion=0.0,
                    recent_abs_motion=0.0,
                    stagnation_score=float(1.0 / (1.0 + theta_abs + 1e-12)),
                )

            def _reconstruct_prune_coordinate_metadata(
                *,
                labels_now: list[str],
                theta_logical_now: np.ndarray,
            ) -> list[ScaffoldCoordinateMetadata]:
                entries: list[ScaffoldCoordinateMetadata] = []
                first_seen_by_label: dict[str, int] = {}
                for step_idx, row in enumerate(history, start=1):
                    if not isinstance(row, dict):
                        continue
                    if row.get("continuation_mode") not in {"phase1_v1", "phase2_v1", "phase3_v1"}:
                        continue
                    labels_step_raw = row.get("selected_ops")
                    labels_step = (
                        [str(x) for x in labels_step_raw]
                        if isinstance(labels_step_raw, Sequence) and not isinstance(labels_step_raw, (str, bytes))
                        else [str(row.get("selected_op", ""))]
                    )
                    positions_step_raw = row.get("selected_positions")
                    positions_step = (
                        [int(x) for x in positions_step_raw]
                        if isinstance(positions_step_raw, Sequence) and not isinstance(positions_step_raw, (str, bytes))
                        else [int(row.get("selected_position", len(entries)))]
                    )
                    feature_rows_raw = row.get("selected_feature_rows")
                    feature_rows = (
                        list(feature_rows_raw)
                        if isinstance(feature_rows_raw, Sequence) and not isinstance(feature_rows_raw, (str, bytes))
                        else []
                    )
                    original_positions_seen: list[int] = []
                    for item_idx, label_step in enumerate(labels_step):
                        pos_orig = (
                            int(positions_step[item_idx])
                            if item_idx < len(positions_step)
                            else int(len(entries))
                        )
                        pos_eff = int(
                            pos_orig + sum(1 for prev in original_positions_seen if int(prev) <= int(pos_orig))
                        )
                        feature_row = feature_rows[item_idx] if item_idx < len(feature_rows) else row
                        feature_mapping = feature_row if isinstance(feature_row, Mapping) else row
                        first_seen_step = int(first_seen_by_label.setdefault(str(label_step), int(step_idx)))
                        meta = ScaffoldCoordinateMetadata(
                            candidate_label=str(label_step),
                            generator_id=(
                                str(feature_mapping.get("generator_id"))
                                if feature_mapping.get("generator_id") is not None
                                else None
                            ),
                            admission_step=int(step_idx),
                            first_seen_step=int(first_seen_step),
                            selector_score=float(_selector_score_value(feature_mapping)),
                            selector_burden=float(_selector_burden_value(feature_mapping)),
                            cooldown_remaining=0,
                            cumulative_abs_motion=0.0,
                            recent_abs_motion=0.0,
                            stagnation_score=0.0,
                        )
                        pos_clamped = max(0, min(int(pos_eff), len(entries)))
                        entries.insert(pos_clamped, meta)
                        original_positions_seen.append(int(pos_orig))
                aligned: list[ScaffoldCoordinateMetadata] = []
                cursor = 0
                theta_vals = np.asarray(theta_logical_now, dtype=float).reshape(-1)
                theta_scale = float(np.median(np.abs(theta_vals))) if theta_vals.size > 0 else 0.0
                theta_scale = float(max(theta_scale, 1e-12))
                for idx_now, label_now in enumerate(labels_now):
                    match_idx: int | None = None
                    for entry_idx in range(cursor, len(entries)):
                        if str(entries[entry_idx].candidate_label) == str(label_now):
                            match_idx = int(entry_idx)
                            break
                    if match_idx is None:
                        base_meta = _fallback_prune_metadata(
                            label=str(label_now),
                            theta_value=(float(theta_vals[idx_now]) if idx_now < int(theta_vals.size) else 0.0),
                            index=int(idx_now),
                        )
                    else:
                        cursor = int(match_idx + 1)
                        base_meta = entries[int(match_idx)]
                    theta_abs = abs(float(theta_vals[idx_now])) if idx_now < int(theta_vals.size) else 0.0
                    stagnation_score = float(max(0.0, 1.0 - theta_abs / theta_scale))
                    aligned.append(
                        ScaffoldCoordinateMetadata(
                            **{
                                **base_meta.__dict__,
                                "stagnation_score": float(stagnation_score),
                            }
                        )
                    )
                return aligned

            def _prune_refit_window_indices(
                *,
                removal_index: int,
                metadata_rows: list[ScaffoldCoordinateMetadata],
                n_plus: int,
            ) -> list[int]:
                n_after = int(max(0, n_plus - 1))
                if n_after <= 0:
                    return []
                omega_eff = int(max(1, min(int(prune_cfg.local_window_size), n_after)))
                local_start = int(max(0, min(int(removal_index) - (omega_eff - 1) // 2, n_after - omega_eff)))
                local_window = list(range(int(local_start), int(local_start + omega_eff)))
                nonlocal_indices = [idx for idx in range(n_after) if idx not in set(local_window)]
                oldest_count = int(math.ceil(float(prune_cfg.old_fraction) * float(len(nonlocal_indices))))
                oldest_count = int(max(0, min(oldest_count, len(nonlocal_indices))))
                if oldest_count > 0:
                    oldest_tail = sorted(
                        nonlocal_indices,
                        key=lambda idx: (
                            int(metadata_rows[idx].admission_step),
                            int(metadata_rows[idx].first_seen_step),
                            str(metadata_rows[idx].candidate_label),
                        ),
                    )[:oldest_count]
                else:
                    oldest_tail = []
                return sorted({int(x) for x in local_window + oldest_tail})

            pre_prune_ops = list(selected_ops)
            pre_prune_layout = _build_selected_layout(pre_prune_ops)
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
            theta_logical_prune_current = np.asarray(_logical_theta_alias(theta, selected_layout), dtype=float)
            prune_metadata = _reconstruct_prune_coordinate_metadata(
                labels_now=[str(op.label) for op in selected_ops],
                theta_logical_now=np.asarray(theta_logical_prune_current, dtype=float),
            )
            candidate_indices = rank_prune_candidates(
                theta=np.asarray(theta_logical_prune_current, dtype=float),
                labels=[str(op.label) for op in selected_ops],
                marginal_proxy_benefit=list(prune_proxy_benefit),
                max_candidates=int(prune_cfg.max_candidates),
                min_candidates=int(prune_cfg.min_candidates),
                fraction_candidates=float(prune_cfg.fraction_candidates),
                selector_burden=[float(meta.selector_burden) for meta in prune_metadata],
                admission_steps=[int(meta.admission_step) for meta in prune_metadata],
                first_seen_steps=[int(meta.first_seen_step) for meta in prune_metadata],
                cooldown_remaining=[int(meta.cooldown_remaining) for meta in prune_metadata],
                stagnation_scores=[float(meta.stagnation_score) for meta in prune_metadata],
                current_step=int(len(history)),
                protect_steps=int(prune_cfg.protect_steps),
                stale_age=int(prune_cfg.stale_age),
                stagnation_threshold=float(prune_cfg.stagnation_threshold),
                small_theta_abs=float(prune_cfg.small_theta_abs),
                small_theta_relative=float(prune_cfg.small_theta_relative),
            )
            prune_summary["permission_open"] = bool(int(len(history)) >= int(prune_cfg.stale_age))
            prune_summary["candidate_count"] = int(len(candidate_indices))
            prune_summary["marginal_proxy_benefit"] = [float(x) for x in prune_proxy_benefit]
            prune_summary["probe_indices"] = [int(x) for x in candidate_indices]
            prune_summary["probe_labels"] = [str(selected_ops[int(i)].label) for i in candidate_indices]
            prune_summary["metadata"] = [dict(meta.__dict__) for meta in prune_metadata]

            def _refit_given_ops(
                ops_refit: list[AnsatzTerm],
                theta0: np.ndarray,
                active_logical_indices: list[int] | None = None,
            ) -> tuple[np.ndarray, float]:
                nonlocal phase2_optimizer_memory
                if len(ops_refit) == 0:
                    return np.zeros(0, dtype=float), float(energy_current)
                layout_refit = _build_selected_layout(ops_refit)
                executor_refit = _build_compiled_executor(ops_refit) if adapt_state_backend_key == "compiled" else None

                def _obj_prune(x: np.ndarray) -> float:
                    return _evaluate_selected_energy_objective(
                        ops_now=ops_refit,
                        theta_now=np.asarray(x, dtype=float),
                        executor_now=executor_refit,
                        parameter_layout_now=layout_refit,
                        objective_stage="prune_refit",
                        depth_marker=int(len(history)),
                    )

                theta_full = np.asarray(theta0, dtype=float).reshape(-1)
                if active_logical_indices is None:
                    active_runtime_indices = list(range(int(layout_refit.runtime_parameter_count)))
                else:
                    active_runtime_indices = runtime_indices_for_logical_indices(
                        layout_refit,
                        [int(i) for i in active_logical_indices],
                    )
                if not active_runtime_indices:
                    return np.asarray(theta_full, dtype=float), float(_obj_prune(theta_full))
                _obj_prune_reduced, opt_x0 = _make_reduced_objective(
                    np.asarray(theta_full, dtype=float),
                    active_runtime_indices,
                    _obj_prune,
                )
                if adapt_inner_optimizer_key == "SPSA":
                    refit_memory = None
                    if phase2_enabled:
                        refit_memory = phase2_memory_adapter.select_active(
                            phase2_optimizer_memory,
                            active_indices=list(active_runtime_indices),
                            source="adapt.post_prune_refit.active_subset",
                        )
                    res = spsa_minimize(
                        fun=_obj_prune_reduced,
                        x0=opt_x0,
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
                    theta_out = np.asarray(theta_full, dtype=float).copy()
                    if len(active_runtime_indices) == int(theta_out.size):
                        theta_out = np.asarray(res.x, dtype=float)
                    else:
                        result_x = np.asarray(res.x, dtype=float).ravel()
                        for k, idx_active in enumerate(active_runtime_indices):
                            theta_out[int(idx_active)] = float(result_x[k])
                    energy_out = float(res.fun)
                    if phase2_enabled:
                        phase2_optimizer_memory = phase2_memory_adapter.merge_active(
                            phase2_optimizer_memory,
                            active_indices=list(active_runtime_indices),
                            active_state=phase2_memory_adapter.from_result(
                                res,
                                method=str(adapt_inner_optimizer_key),
                                parameter_count=int(len(active_runtime_indices)),
                                source="adapt.post_prune_refit.result",
                            ),
                            source="adapt.post_prune_refit.merge",
                        )
                    return np.asarray(theta_out, dtype=float), float(energy_out)
                res = _run_scipy_adapt_optimizer(
                    method_key=str(adapt_inner_optimizer_key),
                    objective=_obj_prune_reduced,
                    x0=opt_x0,
                    maxiter=int(max(25, min(int(maxiter), 120))),
                    context_label="prune refit",
                    scipy_minimize_fn=scipy_minimize,
                )
                theta_out = np.asarray(theta_full, dtype=float).copy()
                if len(active_runtime_indices) == int(theta_out.size):
                    theta_out = np.asarray(res.x, dtype=float)
                else:
                    result_x = np.asarray(res.x, dtype=float).ravel()
                    for k, idx_active in enumerate(active_runtime_indices):
                        theta_out[int(idx_active)] = float(result_x[k])
                return np.asarray(theta_out, dtype=float), float(res.fun)

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

            def _frozen_ablation_energy(
                idx_remove: int,
                theta_cur: np.ndarray,
                labels_cur: list[str],
            ) -> tuple[float, np.ndarray]:
                ops_current = _ops_from_labels(list(labels_cur))
                layout_current = _build_selected_layout(ops_current)
                runtime_remove_indices = runtime_indices_for_logical_indices(layout_current, [int(idx_remove)])
                ops_trial = list(ops_current)
                del ops_trial[int(idx_remove)]
                theta_trial0 = np.delete(np.asarray(theta_cur, dtype=float), runtime_remove_indices)
                executor_trial = (
                    _build_compiled_executor(ops_trial)
                    if adapt_state_backend_key == "compiled" and len(ops_trial) > 0
                    else None
                )
                energy_trial = _evaluate_selected_energy_objective(
                    ops_now=ops_trial,
                    theta_now=np.asarray(theta_trial0, dtype=float),
                    executor_now=executor_trial,
                    parameter_layout_now=_build_selected_layout(ops_trial),
                    objective_stage="prune_frozen",
                    depth_marker=int(len(history)),
                )
                return float(energy_trial), np.asarray(theta_trial0, dtype=float)

            best_candidate_index: int | None = None
            best_candidate_label: str | None = None
            best_trial_window_logical: list[int] = []
            best_frozen_score = float("inf")
            best_frozen_regression = float("inf")
            prune_frozen_rows: list[dict[str, Any]] = []
            for idx_probe in candidate_indices:
                frozen_energy, _theta_frozen = _frozen_ablation_energy(
                    int(idx_probe),
                    np.asarray(theta, dtype=float),
                    [str(op.label) for op in selected_ops],
                )
                frozen_regression = float(frozen_energy - pre_prune_energy)
                selector_burden = (
                    float(prune_metadata[int(idx_probe)].selector_burden)
                    if int(idx_probe) < len(prune_metadata)
                    else 0.0
                )
                cheap_score_prune = cheap_prune_score(
                    frozen_regression=float(frozen_regression),
                    selector_burden=float(selector_burden),
                )
                metadata_after = [
                    meta for meta_idx, meta in enumerate(prune_metadata)
                    if int(meta_idx) != int(idx_probe)
                ]
                prune_window_logical = _prune_refit_window_indices(
                    removal_index=int(idx_probe),
                    metadata_rows=list(metadata_after),
                    n_plus=int(len(pre_prune_ops)),
                )
                prune_frozen_rows.append(
                    {
                        "index": int(idx_probe),
                        "label": str(selected_ops[int(idx_probe)].label),
                        "frozen_energy": float(frozen_energy),
                        "frozen_regression": float(frozen_regression),
                        "selector_burden": float(selector_burden),
                        "cheap_prune_score": float(cheap_score_prune),
                        "refit_window_indices": [int(x) for x in prune_window_logical],
                    }
                )
                candidate_key = (
                    float(cheap_score_prune),
                    float(frozen_regression),
                    int(idx_probe),
                    str(selected_ops[int(idx_probe)].label),
                )
                incumbent_key = (
                    float(best_frozen_score),
                    float(best_frozen_regression),
                    int(best_candidate_index if best_candidate_index is not None else 10**9),
                    str(best_candidate_label or ""),
                )
                if best_candidate_index is None or candidate_key < incumbent_key:
                    best_candidate_index = int(idx_probe)
                    best_candidate_label = str(selected_ops[int(idx_probe)].label)
                    best_trial_window_logical = [int(x) for x in prune_window_logical]
                    best_frozen_score = float(cheap_score_prune)
                    best_frozen_regression = float(frozen_regression)

            prune_summary["frozen_scores"] = [dict(x) for x in prune_frozen_rows]
            prune_summary["selected_index"] = (
                int(best_candidate_index) if best_candidate_index is not None else None
            )
            prune_summary["selected_label"] = (
                str(best_candidate_label) if best_candidate_label is not None else None
            )
            retained_reference_energy = (
                float(history[-1].get("energy_before_opt", pre_prune_energy))
                if history and isinstance(history[-1], Mapping)
                else float(pre_prune_energy)
            )
            admitted_gain = float(max(0.0, retained_reference_energy - float(pre_prune_energy)))

            def _eval_with_removal(
                idx_remove: int,
                theta_cur: np.ndarray,
                labels_cur: list[str],
            ) -> tuple[float, np.ndarray]:
                ops_current = _ops_from_labels(list(labels_cur))
                layout_current = _build_selected_layout(ops_current)
                runtime_remove_indices = runtime_indices_for_logical_indices(layout_current, [int(idx_remove)])
                ops_trial = list(ops_current)
                del ops_trial[int(idx_remove)]
                theta_trial0 = np.delete(np.asarray(theta_cur, dtype=float), runtime_remove_indices)
                theta_trial_opt, e_trial = _refit_given_ops(
                    ops_trial,
                    theta_trial0,
                    active_logical_indices=[int(x) for x in best_trial_window_logical],
                )
                return float(e_trial), np.asarray(theta_trial_opt, dtype=float)

            theta_pruned, labels_pruned, prune_decisions, energy_after_prune = apply_pruning(
                theta=np.asarray(theta, dtype=float),
                labels=[str(op.label) for op in selected_ops],
                candidate_indices=([int(best_candidate_index)] if best_candidate_index is not None else []),
                eval_with_removal=_eval_with_removal,
                energy_before=float(energy_current),
                max_regression=float(prune_cfg.max_regression),
                retained_reference_energy=float(retained_reference_energy),
                admitted_gain=float(admitted_gain),
                retained_gain_ratio=float(prune_cfg.retained_gain_ratio),
            )
            accepted_count = int(sum(1 for d in prune_decisions if bool(d.accepted)))
            retained_gain_after_prune = (
                float(retained_reference_energy - float(energy_after_prune))
                if best_candidate_index is not None
                else None
            )
            prune_summary["trial"] = dict(
                MaturePruneTrial(
                    selector_step=int(len(history)),
                    gate_open=bool(prune_summary.get("permission_open", False)),
                    probe_indices=[int(x) for x in candidate_indices],
                    selected_index=(int(best_candidate_index) if best_candidate_index is not None else None),
                    selected_label=(str(best_candidate_label) if best_candidate_label is not None else None),
                    frozen_regression=(
                        float(best_frozen_regression)
                        if best_candidate_index is not None and math.isfinite(best_frozen_regression)
                        else None
                    ),
                    refit_energy=(
                        float(energy_after_prune)
                        if best_candidate_index is not None
                        else None
                    ),
                    retained_gain=(
                        float(retained_gain_after_prune)
                        if retained_gain_after_prune is not None
                        else None
                    ),
                    accepted=bool(accepted_count > 0),
                    rollback_reason=(
                        str(prune_decisions[0].reason)
                        if prune_decisions and not bool(prune_decisions[0].accepted)
                        else None
                    ),
                ).__dict__
            )
            if accepted_count > 0:
                accepted_remove_indices = [int(d.index) for d in prune_decisions if bool(d.accepted)]
                accepted_runtime_remove_indices = runtime_indices_for_logical_indices(
                    pre_prune_layout,
                    accepted_remove_indices,
                )
                if phase2_enabled:
                    phase2_optimizer_memory = phase2_memory_adapter.remap_remove(
                        phase2_optimizer_memory,
                        indices=list(accepted_runtime_remove_indices),
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
                selected_layout = _build_selected_layout(selected_ops)
                theta = np.asarray(theta_pruned, dtype=float)
                energy_current = float(energy_after_prune)
                if adapt_state_backend_key == "compiled":
                    selected_executor = _build_compiled_executor(selected_ops) if len(selected_ops) > 0 else None
                else:
                    selected_executor = None
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
                        selected_layout = _build_selected_layout(selected_ops)
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
            elif best_candidate_index is not None and int(best_candidate_index) < len(prune_metadata):
                cooled = dict(prune_metadata[int(best_candidate_index)].__dict__)
                cooled["cooldown_remaining"] = int(prune_cfg.cooldown_steps)
                prune_summary["metadata"][int(best_candidate_index)] = cooled

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

        selected_layout = _build_selected_layout(selected_ops)
        theta_logical_final = _logical_theta_alias(theta, selected_layout)

        # Build final state
        if adapt_state_backend_key == "compiled" and len(selected_ops) > 0 and selected_executor is None:
            selected_executor = _build_compiled_executor(selected_ops)
        psi_adapt = _prepare_selected_state(
            ops_now=selected_ops,
            theta_now=theta,
            executor_now=selected_executor,
            parameter_layout_now=selected_layout,
        )
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
        selected_prune_history = [
            _compact_prune_audit(row.get("post_admission_prune"))
            for row in history
            if isinstance(row, Mapping) and isinstance(row.get("post_admission_prune"), Mapping)
        ]
        selected_prune_key = (
            dict(beam_search_diagnostics.get("winner_prune_key", {}))
            if bool(beam_policy.beam_enabled)
            else {
                "energy": float(energy_current),
                "cumulative_selector_score": float(sum(_selector_score_value(row) for row in history)),
                "cumulative_selector_burden": float(sum(_selector_burden_value(row) for row in history)),
                "ansatz_depth": int(len(selected_ops)),
                "labels": [str(op.label) for op in selected_ops],
                "theta_round10": [round(float(x), 10) for x in np.asarray(theta, dtype=float).reshape(-1).tolist()],
                "theta_round10_digits": 10,
                "branch_id": None,
            }
        )
        selected_operator_labels = [str(op.label) for op in selected_ops]
        selected_generator_ids = [
            str(meta.get("generator_id", ""))
            for meta in selected_generator_metadata
            if str(meta.get("generator_id", "")) != ""
        ]
        selected_theta_adapt = [float(x) for x in np.asarray(theta_logical_final, dtype=float).tolist()]
        selected_controller_snapshot = phase1_stage.snapshot().get("last_snapshot")
        selected_controller_snapshot_payload = _controller_snapshot_payload(
            selected_controller_snapshot
        )
        selected_branch_state_summary = _branch_state_summary_payload(
            beam_enabled=bool(beam_policy.beam_enabled),
            branch_id=(
                int(beam_search_diagnostics.get("winner_branch_id"))
                if bool(beam_policy.beam_enabled) and beam_search_diagnostics.get("winner_branch_id") is not None
                else None
            ),
            parent_branch_id=(
                int(beam_search_diagnostics.get("winner_parent_branch_id"))
                if bool(beam_policy.beam_enabled)
                and beam_search_diagnostics.get("winner_parent_branch_id") is not None
                else None
            ),
            history_rows=history,
            depth_local=int(len(history)),
            ansatz_depth=int(len(selected_ops)),
            terminated=(
                bool(beam_search_diagnostics.get("winner_branch_summary", {}).get("terminated", False))
                if bool(beam_policy.beam_enabled)
                else True
            ),
            termination_label=(
                beam_search_diagnostics.get("winner_branch_summary", {}).get("termination_label")
                if bool(beam_policy.beam_enabled)
                else str(stop_reason)
            ),
            cumulative_selector_score=float(
                selected_prune_key.get(
                    "cumulative_selector_score",
                    sum(_selector_score_value(row) for row in history),
                )
            ),
            cumulative_selector_burden=float(
                selected_prune_key.get(
                    "cumulative_selector_burden",
                    sum(_selector_burden_value(row) for row in history),
                )
            ),
            stage_name=str(phase1_stage.stage_name),
            residual_opened=bool(phase1_residual_opened),
            last_probe_reason=str(phase1_last_probe_reason),
            stage_events=phase1_stage_events,
            last_snapshot=selected_controller_snapshot,
        )
        stage_controller_payload = {
            "shortlist_size": int(phase1_shortlist_size_val),
            "plateau_patience": int(phase1_stage_cfg.plateau_patience),
            "probe_margin_ratio": float(phase1_stage_cfg.probe_margin_ratio),
            "max_probe_positions": int(phase1_stage_cfg.max_probe_positions),
            "append_admit_threshold": float(phase1_stage_cfg.append_admit_threshold),
        }
        selected_scaffold_audit = {
            "source_kind": ("beam_winner" if bool(beam_policy.beam_enabled) else "main_branch"),
            "beam_enabled": bool(beam_policy.beam_enabled),
            "branch_id": (
                int(beam_search_diagnostics.get("winner_branch_id"))
                if bool(beam_policy.beam_enabled) and beam_search_diagnostics.get("winner_branch_id") is not None
                else None
            ),
            "parent_branch_id": (
                int(beam_search_diagnostics.get("winner_parent_branch_id"))
                if bool(beam_policy.beam_enabled) and beam_search_diagnostics.get("winner_parent_branch_id") is not None
                else None
            ),
            "depth_local": int(len(history)),
            "stop_reason": str(stop_reason),
            "energy": float(energy_current),
            "operators": list(selected_operator_labels),
            "generator_ids": list(selected_generator_ids),
            "prune_key": dict(selected_prune_key),
            "last_prune": _compact_prune_audit(prune_summary),
            "prune_history": [dict(x) for x in selected_prune_history],
            "phase3_surface_summary": _phase3_surface_audit_payload(
                scored_rows=phase2_last_shortlist_records,
                retained_rows=phase2_last_retained_shortlist_records,
                admitted_rows=phase2_last_admitted_records,
                beam_enabled=bool(beam_policy.beam_enabled),
            ),
            "stage_events": [dict(row) for row in phase1_stage_events],
            "last_probe_reason": str(phase1_last_probe_reason),
            "residual_opened": bool(phase1_residual_opened),
            "branch_state_summary": dict(selected_branch_state_summary),
        }
        selected_scaffold_history: list[dict[str, Any]] = []
        for step_idx, row in enumerate(history, start=1):
            if not isinstance(row, Mapping):
                continue
            labels_step_raw = row.get("selected_ops")
            labels_step = (
                [str(x) for x in labels_step_raw]
                if isinstance(labels_step_raw, Sequence) and not isinstance(labels_step_raw, (str, bytes))
                else [str(row.get("selected_op", ""))]
            )
            positions_step_raw = row.get("selected_positions")
            positions_step = (
                [int(x) for x in positions_step_raw]
                if isinstance(positions_step_raw, Sequence) and not isinstance(positions_step_raw, (str, bytes))
                else [int(row.get("selected_position", 0))]
            )
            feature_rows_raw = row.get("selected_feature_rows")
            feature_rows = (
                list(feature_rows_raw)
                if isinstance(feature_rows_raw, Sequence) and not isinstance(feature_rows_raw, (str, bytes))
                else []
            )
            selected_records_step: list[dict[str, Any]] = []
            for item_idx, label_step in enumerate(labels_step):
                feature_mapping = (
                    feature_rows[item_idx]
                    if item_idx < len(feature_rows) and isinstance(feature_rows[item_idx], Mapping)
                    else row
                )
                selected_records_step.append(
                    {
                        "generator_label": str(label_step),
                        "position_id": (
                            int(positions_step[item_idx])
                            if item_idx < len(positions_step)
                            else int(row.get("selected_position", 0))
                        ),
                        "generator_id": (
                            str(feature_mapping.get("generator_id"))
                            if feature_mapping.get("generator_id") is not None
                            else None
                        ),
                        "template_id": (
                            str(feature_mapping.get("template_id"))
                            if feature_mapping.get("template_id") is not None
                            else None
                        ),
                        "selection_mode": str(row.get("selection_mode", "")),
                        "runtime_split_mode": str(
                            feature_mapping.get("runtime_split_mode", row.get("runtime_split_mode", "off"))
                        ),
                    }
                )
            selected_scaffold_history.append(
                {
                    "step_index": int(row.get("depth", step_idx)),
                    "batch_selected": bool(row.get("batch_selected", False)),
                    "beam_structural_mode": (
                        None
                        if row.get("beam_structural_mode") is None
                        else str(row.get("beam_structural_mode"))
                    ),
                    "selection_mode": str(row.get("selection_mode", "")),
                    "energy_after_opt": float(row.get("energy_after_opt", energy_current)),
                    "selected_records": selected_records_step,
                    "post_admission_prune": _compact_prune_audit(row.get("post_admission_prune")),
                }
            )
        selected_scaffold_record_chain: list[dict[str, Any]] = []
        for step in selected_scaffold_history:
            step_index = int(step.get("step_index", len(selected_scaffold_record_chain) + 1))
            selected_records_raw = step.get("selected_records", [])
            selected_records_step = (
                list(selected_records_raw)
                if isinstance(selected_records_raw, Sequence) and not isinstance(selected_records_raw, (str, bytes))
                else []
            )
            for record_idx, record in enumerate(selected_records_step, start=1):
                if not isinstance(record, Mapping):
                    continue
                selected_scaffold_record_chain.append(
                    {
                        "history_label": "H_*",
                        "step_index": int(step_index),
                        "record_index": int(record_idx),
                        "beam_structural_mode": step.get("beam_structural_mode"),
                        "selection_mode": str(step.get("selection_mode", "")),
                        "energy_after_opt": float(step.get("energy_after_opt", energy_current)),
                        "generator_label": str(record.get("generator_label", "")),
                        "position_id": int(record.get("position_id", 0)),
                        "generator_id": (
                            str(record.get("generator_id"))
                            if record.get("generator_id") is not None
                            else None
                        ),
                        "template_id": (
                            str(record.get("template_id"))
                            if record.get("template_id") is not None
                            else None
                        ),
                        "runtime_split_mode": str(record.get("runtime_split_mode", "off")),
                    }
                )
        selected_scaffold_last_step = (
            selected_scaffold_history[-1]
            if selected_scaffold_history and isinstance(selected_scaffold_history[-1], Mapping)
            else None
        )
        final_choice_summary: dict[str, Any]
        if bool(beam_policy.beam_enabled):
            winner_branch_summary = (
                dict(beam_search_diagnostics.get("winner_branch_summary", {}))
                if isinstance(beam_search_diagnostics.get("winner_branch_summary"), Mapping)
                else {}
            )
            transition_kind = str(winner_branch_summary.get("last_transition_kind", "unknown"))
            admitted_count = int(winner_branch_summary.get("last_admission_record_count", 0) or 0)
            final_choice_summary = {
                "selection_source": str(selected_scaffold_audit["source_kind"]),
                "beam_enabled": True,
                "structural_choice": (
                    "stop"
                    if transition_kind == "stop_child"
                    else ("admit" if transition_kind == "admission_child" else "none")
                ),
                "transition_kind": str(transition_kind),
                "beam_child_kind": (
                    "stop_child"
                    if transition_kind == "stop_child"
                    else ("non_stop_child" if transition_kind == "admission_child" else str(transition_kind))
                ),
                "admission_kind": (
                    "none"
                    if admitted_count <= 0
                    else (
                        "reduced_plane_batch_admission"
                        if admitted_count > 1
                        else "singleton_admission"
                    )
                ),
                "selected_record_count": int(admitted_count),
                "batch_selected": bool(admitted_count > 1),
                "branch_terminated": bool(winner_branch_summary.get("terminated", False)),
                "branch_stop_reason": (
                    None
                    if winner_branch_summary.get("stop_reason") is None
                    else str(winner_branch_summary.get("stop_reason"))
                ),
                "step_index": (
                    int(selected_scaffold_last_step.get("step_index"))
                    if transition_kind == "admission_child" and isinstance(selected_scaffold_last_step, Mapping)
                    else None
                ),
                "selection_mode": (
                    str(selected_scaffold_last_step.get("selection_mode", ""))
                    if transition_kind == "admission_child" and isinstance(selected_scaffold_last_step, Mapping)
                    else None
                ),
                "beam_structural_mode": (
                    selected_scaffold_last_step.get("beam_structural_mode")
                    if transition_kind == "admission_child" and isinstance(selected_scaffold_last_step, Mapping)
                    else None
                ),
            }
        else:
            admitted_count = (
                len(selected_scaffold_last_step.get("selected_records", []))
                if isinstance(selected_scaffold_last_step, Mapping)
                and isinstance(selected_scaffold_last_step.get("selected_records"), Sequence)
                else 0
            )
            batch_selected = bool(
                isinstance(selected_scaffold_last_step, Mapping)
                and selected_scaffold_last_step.get("batch_selected", False)
            )
            final_choice_summary = {
                "selection_source": str(selected_scaffold_audit["source_kind"]),
                "beam_enabled": False,
                "structural_choice": ("admit" if admitted_count > 0 else "none"),
                "transition_kind": ("main_path_admission" if admitted_count > 0 else "none"),
                "beam_child_kind": None,
                "admission_kind": (
                    "none"
                    if admitted_count <= 0
                    else (
                        "reduced_plane_batch_admission"
                        if batch_selected or admitted_count > 1
                        else "singleton_admission"
                    )
                ),
                "selected_record_count": int(admitted_count),
                "batch_selected": bool(batch_selected or admitted_count > 1),
                "branch_terminated": None,
                "branch_stop_reason": str(stop_reason),
                "step_index": (
                    int(selected_scaffold_last_step.get("step_index"))
                    if isinstance(selected_scaffold_last_step, Mapping)
                    else None
                ),
                "selection_mode": (
                    str(selected_scaffold_last_step.get("selection_mode", ""))
                    if isinstance(selected_scaffold_last_step, Mapping)
                    else None
                ),
                "beam_structural_mode": (
                    selected_scaffold_last_step.get("beam_structural_mode")
                    if isinstance(selected_scaffold_last_step, Mapping)
                    else None
                ),
            }
        selected_scaffold_summary = {
            "selection_source": str(selected_scaffold_audit["source_kind"]),
            "scaffold_label": "O_*",
            "theta_label": "theta_*^adapt",
            "history_label": "H_*",
            "manifold_label": "M_scaf(O_*)",
            "operator_labels": list(selected_operator_labels),
            "theta_adapt": list(selected_theta_adapt),
            "generator_ids": list(selected_generator_ids),
            "ansatz_depth": int(len(selected_ops)),
            "manifold_dimension": int(len(selected_ops)),
            "history_step_count": int(len(selected_scaffold_history)),
            "history_record_count": int(len(selected_scaffold_record_chain)),
            "history_record_chain_label": "H_*",
            "beam_enabled": bool(beam_policy.beam_enabled),
            "branch_id": selected_scaffold_audit.get("branch_id"),
            "final_choice_summary": dict(final_choice_summary),
            "branch_state_summary": dict(selected_branch_state_summary),
        }
        selected_scaffold_audit["final_choice_summary"] = dict(final_choice_summary)
        selected_state_summary = {
            "state_label": "|psi_*>",
            "state_preparation_label": "U(theta_*^adapt; O_*)|phi_0>",
            "reference_state_label": "|phi_0>",
            "scaffold_label": "O_*",
            "theta_label": "theta_*^adapt",
            "manifold_label": "M_scaf(O_*)",
            "coordinate_space_notation": "R^{|O_*|}",
            "ansatz_depth": int(len(selected_ops)),
            "manifold_dimension": int(len(selected_ops)),
            "beam_enabled": bool(beam_policy.beam_enabled),
            "branch_id": selected_scaffold_audit.get("branch_id"),
            "state_norm": float(np.linalg.norm(np.asarray(psi_adapt, dtype=complex))),
        }
        selected_memory_contract_summary = _optimizer_memory_contract_summary_payload(
            beam_enabled=bool(beam_policy.beam_enabled),
            branch_id=selected_scaffold_audit.get("branch_id"),
            memory_state=phase2_optimizer_memory,
            operator_labels=selected_operator_labels,
            generator_ids=selected_generator_ids,
            num_parameters=int(theta.size),
            last_active_subset_source=str(phase2_last_optimizer_memory_source),
            last_active_subset_reused=bool(phase2_last_optimizer_memory_reused),
        )
        selected_scaffold_summary["selected_state_summary"] = dict(selected_state_summary)
        selected_scaffold_summary["optimizer_memory_contract_summary"] = dict(
            selected_memory_contract_summary
        )
        selected_scaffold_audit["selected_state_summary"] = dict(selected_state_summary)
        selected_scaffold_audit["optimizer_memory_contract_summary"] = dict(
            selected_memory_contract_summary
        )
        controller_runtime_boundary_summary = _controller_runtime_boundary_summary_payload(
            phase_enabled=bool(phase1_enabled),
            cfg=phase1_stage_cfg,
            stage_controller_payload=stage_controller_payload,
            current_snapshot_payload=selected_controller_snapshot_payload,
            beam_enabled=bool(beam_policy.beam_enabled),
            branch_id=selected_scaffold_audit.get("branch_id"),
        )
        active_phase3_surface_summary: dict[str, Any] | None = None
        active_hh_pool_summary: dict[str, Any] | None = None
        if phase3_enabled:
            phase3_scored_rows = [dict(row) for row in phase2_last_shortlist_records[-200:]]
            phase3_retained_rows = [
                dict(row) for row in phase2_last_retained_shortlist_records[-200:]
            ]
            phase3_admitted_rows = [dict(row) for row in phase2_last_admitted_records[-200:]]
            active_phase3_surface_summary = {
                "surface_label": "Omega_HH^(3)",
                "source_rows_key": "phase2_shortlist_rows",
                "source_row_semantics": "last_scored_candidate_surface",
                "scored_rows_key": "phase2_scored_rows",
                "scored_rows_semantics": "last_scored_candidate_surface",
                "retained_rows_key": "phase2_retained_shortlist_rows",
                "retained_rows_semantics": "controller_retained_shortlist",
                "admitted_rows_key": "phase2_admitted_rows",
                "admitted_rows_semantics": "reduced_plane_admitted_set",
                "candidate_count": int(len(phase3_scored_rows)),
                "retained_shortlist_count": int(len(phase3_retained_rows)),
                "admitted_count": int(len(phase3_admitted_rows)),
                "phase2_shortlisted_count": int(
                    sum(bool(row.get("phase2_shortlisted", False)) for row in phase3_scored_rows)
                ),
                "phase3_shortlisted_count": int(
                    sum(bool(row.get("phase3_shortlisted", False)) for row in phase3_scored_rows)
                ),
                "operator_labels": list(
                    dict.fromkeys(
                        str(row.get("candidate_label", ""))
                        for row in phase3_scored_rows
                        if str(row.get("candidate_label", "")) != ""
                    )
                ),
                "generator_ids": list(
                    dict.fromkeys(
                        str(row.get("generator_id", ""))
                        for row in phase3_scored_rows
                        if str(row.get("generator_id", "")) != ""
                    )
                ),
                "position_ids": list(
                    dict.fromkeys(
                        int(row.get("position_id"))
                        for row in phase3_scored_rows
                        if row.get("position_id") is not None
                    )
                ),
                "runtime_split_modes": list(
                    dict.fromkeys(
                        str(row.get("runtime_split_mode", "off"))
                        for row in phase3_scored_rows
                    )
                ),
                "score_version": str(phase2_score_cfg.score_version),
                "batch_selected": bool(phase2_last_batch_selected),
                "compatibility_penalty_total": float(phase2_last_batch_penalty_total),
                "selected_operator_labels": list(selected_operator_labels),
                "selected_generator_ids": list(selected_generator_ids),
            }
            active_hh_pool_summary = _active_hh_pool_summary_payload(
                phase1_rows=phase1_last_retained_records,
                phase2_rows=phase2_last_geometric_shortlist_records,
                phase3_rows=phase2_last_retained_shortlist_records,
            )
        exact_energy_from_final_state, _ = energy_via_one_apply(psi_adapt, h_compiled)
        exact_energy_from_final_state = float(exact_energy_from_final_state)
        final_noise_audit_payload: dict[str, Any] | None = None
        if final_noise_audit_config is not None:
            final_noise_audit_snapshot = FinalNoiseAuditSnapshot(
                h_poly=h_poly,
                parameter_layout=selected_layout,
                theta_runtime=tuple(float(x) for x in np.asarray(theta, dtype=float).tolist()),
                theta_logical=tuple(
                    float(x) for x in np.asarray(theta_logical_final, dtype=float).tolist()
                ),
                reference_state=np.array(psi_ref, dtype=complex, copy=True),
                num_qubits=int(phase3_oracle_num_qubits),
                operator_labels=tuple(str(op.label) for op in selected_ops),
                ansatz_depth=int(len(selected_ops)),
                runtime_parameter_count=int(theta.size),
                logical_parameter_count=int(len(selected_ops)),
                exact_filtered_ground_energy=float(exact_gs),
                exact_final_state_energy=float(exact_energy_from_final_state),
            )
            audit_t0 = time.perf_counter()
            _ai_log(
                "hardcoded_adapt_final_noise_audit_start",
                noise_mode=str(final_noise_audit_config.noise_mode),
                mitigation_mode=str(final_noise_audit_config.mitigation_mode),
                strict=bool(final_noise_audit_config.strict),
            )
            try:
                final_noise_audit_payload = _run_final_noise_audit(
                    final_noise_audit_snapshot,
                    final_noise_audit_config,
                )
                _ai_log(
                    "hardcoded_adapt_final_noise_audit_done",
                    status=str(final_noise_audit_payload.get("status", "completed")),
                    elapsed_s=float(time.perf_counter() - audit_t0),
                    exact_target_abs_error=float(
                        final_noise_audit_payload.get("deltas", {}).get(
                            "exact_target_abs_error",
                            float("nan"),
                        )
                    ),
                )
            except Exception as exc:
                _ai_log(
                    "hardcoded_adapt_final_noise_audit_error",
                    elapsed_s=float(time.perf_counter() - audit_t0),
                    strict=bool(final_noise_audit_config.strict),
                    error_type=str(type(exc).__name__),
                    error_repr=repr(exc),
                )
                if bool(final_noise_audit_config.strict):
                    raise
                final_noise_audit_payload = {
                    "status": "failed",
                    "strict": bool(final_noise_audit_config.strict),
                    "requested_config": dict(
                        _final_noise_audit_config_payload(final_noise_audit_config) or {}
                    ),
                    "reference": {
                        "primary_metric_name": "exact_target_abs_error",
                        "exact_filtered_ground_energy": float(exact_gs),
                        "exact_final_state_energy": float(exact_energy_from_final_state),
                    },
                    "failure": {
                        "error_type": str(type(exc).__name__),
                        "error_message": str(exc),
                    },
                }
        phase3_output_motif_library = (
            extract_motif_library(
                generator_metadata=selected_generator_metadata,
                theta=[float(x) for x in np.asarray(theta_logical_final, dtype=float).tolist()],
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
            "gradient_uncertainty_source": str(phase3_gradient_uncertainty_source),
            "oracle_gradient_scope": (
                str(phase3_oracle_gradient_config.scope)
                if phase3_oracle_gradient_config is not None
                else "off"
            ),
            "oracle_gradient_config": _phase3_oracle_gradient_config_payload(phase3_oracle_gradient_config),
            "oracle_execution_surface": (
                str(phase3_oracle_gradient_config.execution_surface)
                if phase3_oracle_gradient_config is not None
                else "off"
            ),
            "oracle_backend_info": (
                dict(phase3_oracle_backend_info)
                if isinstance(phase3_oracle_backend_info, Mapping)
                else phase3_oracle_backend_info
            ),
            "last_oracle_gradient_backend_info": (
                dict(phase3_last_oracle_gradient_backend_info)
                if isinstance(phase3_last_oracle_gradient_backend_info, Mapping)
                else phase3_last_oracle_gradient_backend_info
            ),
            "last_oracle_inner_objective_backend_info": (
                dict(phase3_oracle_inner_backend_info)
                if isinstance(phase3_oracle_inner_backend_info, Mapping)
                else phase3_oracle_inner_backend_info
            ),
            "oracle_raw_transport": (
                str(phase3_oracle_raw_transport)
                if phase3_oracle_raw_transport not in {None, ""}
                else None
            ),
            "oracle_gradient_raw_records_total": int(phase3_oracle_gradient_raw_records_total),
            "oracle_symmetry_diagnostic_calls_total": int(
                phase3_oracle_symmetry_diagnostic_calls_total
            ),
            "oracle_symmetry_diagnostic_raw_records_total": int(
                phase3_oracle_symmetry_diagnostic_raw_records_total
            ),
            "oracle_gradient_raw_artifact_path": (
                None
                if phase3_oracle_raw_artifact_path in {None, ""}
                else str(phase3_oracle_raw_artifact_path)
            ),
            "oracle_gradient_calls_total": int(phase3_oracle_gradient_calls_total),
            "oracle_inner_objective_mode": str(phase3_oracle_inner_objective_mode_key),
            "oracle_inner_objective_mode_requested": str(
                phase3_oracle_inner_objective_mode_requested_key
            ),
            "oracle_inner_objective_runtime_guard_reason": (
                None
                if phase3_oracle_inner_objective_runtime_guard_reason in {None, ""}
                else str(phase3_oracle_inner_objective_runtime_guard_reason)
            ),
            "oracle_inner_objective_calls_total": int(phase3_oracle_inner_objective_calls_total),
            "oracle_inner_objective_raw_records_total": int(
                phase3_oracle_inner_objective_raw_records_total
            ),
            "reoptimization_backend": str(phase3_oracle_inner_backend_name),
            "phase3_enable_rescue_requested": bool(phase3_enable_rescue_requested),
            "phase3_enable_rescue_effective": bool(phase3_enable_rescue_effective),
            "stage_controller": dict(stage_controller_payload),
            "stage_events": [dict(row) for row in phase1_stage_events],
            "phase1_feature_rows": [dict(row) for row in phase1_features_history[-200:]],
            "last_probe_reason": str(phase1_last_probe_reason),
            "residual_opened": bool(phase1_residual_opened),
            "selected_scaffold_summary": dict(selected_scaffold_summary),
            "selected_scaffold_final_choice": dict(final_choice_summary),
            "selected_scaffold_branch_state": dict(selected_branch_state_summary),
            "selected_state_summary": dict(selected_state_summary),
            "selected_scaffold_optimizer_memory_contract": dict(
                selected_memory_contract_summary
            ),
            "controller_runtime_boundary_summary": dict(
                controller_runtime_boundary_summary
            ),
            "selected_scaffold_history": [dict(row) for row in selected_scaffold_history],
            "selected_scaffold_record_chain": [dict(row) for row in selected_scaffold_record_chain],
            "selected_scaffold_audit": dict(selected_scaffold_audit),
            "beam_search": copy.deepcopy(beam_search_diagnostics),
        }
        backend_compile_summary: dict[str, Any] | None = None
        if backend_compile_oracle is not None:
            backend_compile_summary = {
                "mode": str(phase3_backend_cost_mode_key),
                "requested_backend_name": (
                    None if phase3_backend_name in {None, ""} else str(phase3_backend_name)
                ),
                "requested_backend_shortlist": [str(x) for x in phase3_backend_shortlist_tokens],
                "optimization_level": int(phase3_backend_optimization_level),
                "seed_transpiler": int(phase3_backend_transpile_seed),
                "resolution_audit": [dict(row) for row in backend_compile_oracle.resolution_audit],
                "cache_summary": dict(backend_compile_oracle.cache_summary()),
                **dict(backend_compile_oracle.final_scaffold_summary(selected_ops)),
            }
        if phase2_enabled:
            continuation_payload.update(
                {
                    "phase1_retained_rows": [dict(row) for row in phase1_last_retained_records[-200:]],
                    "phase2_shortlist_rows": [dict(row) for row in phase2_last_shortlist_records[-200:]],
                    "phase2_scored_rows": [dict(row) for row in phase2_last_shortlist_records[-200:]],
                    "phase2_geometric_shortlist_rows": [
                        dict(row) for row in phase2_last_geometric_shortlist_records[-200:]
                    ],
                    "phase2_retained_shortlist_rows": [
                        dict(row) for row in phase2_last_retained_shortlist_records[-200:]
                    ],
                    "phase2_admitted_rows": [dict(row) for row in phase2_last_admitted_records[-200:]],
                    "active_hh_pool_summary": (
                        dict(active_hh_pool_summary)
                        if isinstance(active_hh_pool_summary, Mapping)
                        else None
                    ),
                    "optimizer_memory": dict(phase2_optimizer_memory),
                    "phase1": {
                        "lambda_F": float(phase1_score_cfg.lambda_F),
                        "lambda_compile": float(phase1_score_cfg.lambda_compile),
                        "lambda_measure": float(phase1_score_cfg.lambda_measure),
                        "lambda_leak": float(phase1_score_cfg.lambda_leak),
                        "score_z_alpha": float(phase1_score_cfg.z_alpha),
                        "depth_ref": float(phase1_score_cfg.depth_ref),
                        "group_ref": float(phase1_score_cfg.group_ref),
                        "shot_ref": float(phase1_score_cfg.shot_ref),
                        "family_ref": float(phase1_score_cfg.family_ref),
                    },
                    "phase2": {
                        "lambda_F": float(phase2_score_cfg.lambda_F),
                        "score_z_alpha": float(phase2_score_cfg.z_alpha),
                        "shortlist_fraction": float(phase2_score_cfg.shortlist_fraction),
                        "shortlist_size": int(phase2_score_cfg.shortlist_size),
                        "w_depth": float(phase2_score_cfg.wD),
                        "w_group": float(phase2_score_cfg.wG),
                        "w_shot": float(phase2_score_cfg.wC),
                        "w_optdim": float(phase2_score_cfg.wP),
                        "w_reuse": float(phase2_score_cfg.wc),
                        "w_lifetime": float(phase2_score_cfg.lifetime_weight),
                        "eta_L": float(phase2_score_cfg.eta_L),
                        "depth_ref": float(phase2_score_cfg.depth_ref),
                        "group_ref": float(phase2_score_cfg.group_ref),
                        "shot_ref": float(phase2_score_cfg.shot_ref),
                        "optdim_ref": float(phase2_score_cfg.optdim_ref),
                        "reuse_ref": float(phase2_score_cfg.reuse_ref),
                        "family_ref": float(phase2_score_cfg.family_ref),
                        "novelty_eps": float(phase2_score_cfg.novelty_eps),
                        "cheap_score_eps": float(phase2_score_cfg.cheap_score_eps),
                        "metric_floor": float(phase2_score_cfg.metric_floor),
                        "reduced_metric_collapse_rel_tol": float(
                            phase2_score_cfg.reduced_metric_collapse_rel_tol
                        ),
                        "ridge_growth_factor": float(phase2_score_cfg.ridge_growth_factor),
                        "ridge_max_steps": int(phase2_score_cfg.ridge_max_steps),
                        "leakage_cap": float(phase2_score_cfg.leakage_cap),
                        "motif_bonus_weight": float(phase2_score_cfg.motif_bonus_weight),
                        "duplicate_penalty_weight": float(phase2_score_cfg.duplicate_penalty_weight),
                        "frontier_ratio": float(phase2_score_cfg.phase2_frontier_ratio),
                        "phase3_frontier_ratio": float(phase2_score_cfg.phase3_frontier_ratio),
                        "batch_target_size": int(phase2_score_cfg.batch_target_size),
                        "batch_size_cap": int(phase2_score_cfg.batch_size_cap),
                        "batch_near_degenerate_ratio": float(phase2_score_cfg.batch_near_degenerate_ratio),
                        "batch_rank_rel_tol": float(phase2_score_cfg.batch_rank_rel_tol),
                        "batch_additivity_tol": float(phase2_score_cfg.batch_additivity_tol),
                        "compat_overlap_weight": float(phase2_score_cfg.compat_overlap_weight),
                        "compat_comm_weight": float(phase2_score_cfg.compat_comm_weight),
                        "compat_curv_weight": float(phase2_score_cfg.compat_curv_weight),
                        "compat_sched_weight": float(phase2_score_cfg.compat_sched_weight),
                        "compat_measure_weight": float(phase2_score_cfg.compat_measure_weight),
                        "remaining_evaluations_proxy_mode": str(
                            phase2_score_cfg.remaining_evaluations_proxy_mode
                        ),
                    },
                }
            )
        if phase3_enabled:
            continuation_payload.update(
                {
                    "selected_generator_metadata": [dict(x) for x in selected_generator_metadata],
                    "active_phase3_surface_summary": (
                        dict(active_phase3_surface_summary)
                        if isinstance(active_phase3_surface_summary, Mapping)
                        else None
                    ),
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
        if backend_compile_summary is not None:
            continuation_payload["backend_compile_cost_summary"] = dict(backend_compile_summary)

        exact_state_fidelity: float | None = None
        exact_state_fidelity_source: str | None = None
        if phase3_exact_reference_state is not None:
            psi_exact_ref = _normalize_state(np.asarray(phase3_exact_reference_state, dtype=complex).reshape(-1))
            psi_final = _normalize_state(np.asarray(psi_adapt, dtype=complex).reshape(-1))
            if int(psi_exact_ref.size) != int(psi_final.size):
                raise ValueError(
                    "phase3 exact reference state dimension mismatch: "
                    f"got {psi_exact_ref.size}, expected {psi_final.size}."
                )
            exact_state_fidelity = float(abs(np.vdot(psi_exact_ref, psi_final)) ** 2)
            exact_state_fidelity_source = (
                "final_theta_exact_state_sidecar"
                if phase3_oracle_inner_objective_enabled
                else "phase3_rescue_exact_state"
            )

        payload = {
            "success": True,
            "method": method_name,
            "energy": float(energy_current),
            "energy_source": str(phase3_oracle_inner_backend_name),
            "analytic_noise_applied": bool(
                adapt_analytic_noise_enabled
                and (
                    (not phase3_oracle_inner_objective_enabled)
                    or (not phase3_oracle_gradient_enabled)
                )
            ),
            "analytic_noise_energy_path_active": bool(
                adapt_analytic_noise_enabled
                and (not phase3_oracle_inner_objective_enabled)
            ),
            "analytic_noise_gradient_path_active": bool(
                adapt_analytic_noise_enabled
                and (not phase3_oracle_gradient_enabled)
            ),
            "analytic_noise_std": float(adapt_analytic_noise_std_val),
            "analytic_noise_seed": adapt_analytic_noise_seed_val,
            "exact_energy_from_final_state": float(exact_energy_from_final_state),
            "exact_gs_energy": float(exact_gs),
            "delta_e": float(energy_current - exact_gs),
            "abs_delta_e": float(abs(energy_current - exact_gs)),
            "exact_delta_e_from_final_state": float(exact_energy_from_final_state - exact_gs),
            "exact_abs_delta_e_from_final_state": float(abs(exact_energy_from_final_state - exact_gs)),
            "num_particles": {"n_up": int(num_particles[0]), "n_dn": int(num_particles[1])},
            "ansatz_depth": int(len(selected_ops)),
            "num_parameters": int(theta.size),
            "logical_num_parameters": int(len(selected_ops)),
            "optimal_point": [float(x) for x in theta.tolist()],
            "logical_optimal_point": [float(x) for x in theta_logical_final.tolist()],
            "parameterization": serialize_layout(selected_layout),
            "operators": [str(op.label) for op in selected_ops],
            "pool_size": int(len(pool)),
            "pool_type": str(pool_key),
            "adapt_pool_class_filter_json": (
                str(adapt_pool_class_filter_json) if adapt_pool_class_filter_json is not None else None
            ),
            "adapt_pool_class_filter_classifier_version": (
                str(full_meta_class_filter_spec.classifier_version)
                if full_meta_class_filter_spec is not None
                else None
            ),
            "adapt_pool_class_filter_keep_classes": (
                list(full_meta_class_filter_spec.keep_classes)
                if full_meta_class_filter_spec is not None
                else None
            ),
            "adapt_pool_class_filter_class_counts_before": (
                dict(full_meta_class_filter_meta.get("class_counts_before", {}))
                if full_meta_class_filter_meta is not None
                else None
            ),
            "adapt_pool_class_filter_class_counts_after": (
                dict(full_meta_class_filter_meta.get("class_counts_after", {}))
                if full_meta_class_filter_meta is not None
                else None
            ),
            "phase3_oracle_inner_objective_mode": str(phase3_oracle_inner_objective_mode_key),
            "phase3_oracle_inner_objective_mode_requested": str(
                phase3_oracle_inner_objective_mode_requested_key
            ),
            "phase3_oracle_inner_objective_runtime_guard_reason": (
                None
                if phase3_oracle_inner_objective_runtime_guard_reason in {None, ""}
                else str(phase3_oracle_inner_objective_runtime_guard_reason)
            ),
            "exact_state_fidelity": (
                float(exact_state_fidelity) if exact_state_fidelity is not None else None
            ),
            "exact_state_fidelity_source": (
                str(exact_state_fidelity_source) if exact_state_fidelity_source is not None else None
            ),
            "phase1_depth0_full_meta_override": bool(phase1_depth0_full_meta_override),
            "stop_reason": str(stop_reason),
            "nfev_total": int(nfev_total),
            "adapt_inner_optimizer": str(adapt_inner_optimizer_key),
            "adapt_reopt_policy": str(adapt_reopt_policy_key),
            "adapt_window_size": int(adapt_window_size_val),
            "adapt_window_topk": int(adapt_window_topk_val),
            "adapt_full_refit_every": int(adapt_full_refit_every_val),
            "adapt_final_full_refit": bool(adapt_final_full_refit_val),
            "adapt_beam_live_branches_requested": int(beam_policy.live_branches_requested),
            "adapt_beam_children_per_parent_requested": (
                int(beam_policy.children_per_parent_requested)
                if beam_policy.children_per_parent_requested is not None
                else None
            ),
            "adapt_beam_terminated_keep_requested": (
                int(beam_policy.terminated_keep_requested)
                if beam_policy.terminated_keep_requested is not None
                else None
            ),
            "adapt_beam_live_branches": int(beam_policy.live_branches_effective),
            "adapt_beam_children_per_parent": int(beam_policy.children_per_parent_effective),
            "adapt_beam_terminated_keep": int(beam_policy.terminated_keep_effective),
            "adapt_beam_enabled": bool(beam_policy.beam_enabled),
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
        if seq2p_logical_mode:
            payload.update(
                {
                    "logical_parameterization": "double_sequential",
                    "logical_operator_count": int(len(history)),
                    "logical_operator_labels": [
                        str(row.get("selected_logical_op", row.get("selected_op", "")))
                        for row in history
                    ],
                    "logical_operator_sizes": [
                        int(row.get("selected_logical_size", 1))
                        for row in history
                    ],
                    "expanded_operator_count": int(len(selected_ops)),
                }
            )
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
                            "cx_proxy_total",
                            "sq_proxy_total",
                            "gate_proxy_total",
                            "max_pauli_weight",
                            "position_shift_span",
                            "refit_active_count",
                        ],
                    },
                    "compile_cost_mode": str(phase3_backend_cost_mode_key),
                    "backend_compile_cost_summary": (
                        dict(backend_compile_summary) if backend_compile_summary is not None else None
                    ),
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
                        compile_cost_mode=str(phase3_backend_cost_mode_key),
                        backend_target_names=(
                            list(
                                dict.fromkeys(
                                    [
                                        str(row.get("transpile_backend", ""))
                                        for row in (backend_compile_summary or {}).get("rows", [])
                                        if str(row.get("transpile_backend", "")) != ""
                                    ]
                                )
                            )
                            if isinstance(backend_compile_summary, Mapping)
                            else []
                        ),
                        backend_reduction_mode=(
                            "single_backend"
                            if str(phase3_backend_cost_mode_key) == "transpile_single_v1"
                            else (
                                "best_backend_in_shortlist_v1"
                                if str(phase3_backend_cost_mode_key) == "transpile_shortlist_v1"
                                else "none"
                            )
                        ),
                    ).__dict__,
                }
            )
        if adapt_inner_optimizer_key == "SPSA":
            payload["adapt_spsa"] = dict(adapt_spsa_params)
        if final_noise_audit_payload is not None:
            payload["final_noise_audit_v1"] = dict(final_noise_audit_payload)
        if adapt_ref_import is not None:
            payload["adapt_ref_import"] = dict(adapt_ref_import)

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
    finally:
        if phase3_oracle_cleanup is not None:
            phase3_oracle_cleanup.close()
        else:
            _close_phase3_oracle_resource(phase3_oracle)


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
    require_matplotlib()
    settings = payload.get("settings", {})
    adapt = payload.get("adapt_vqe", {})
    problem = settings.get("problem", "hubbard")
    model_name = "Hubbard-Holstein" if problem == "hh" else "Hubbard"

    manifest_sections: list[tuple[str, list[tuple[str, Any]]]] = [
        (
            "Model and regime",
            [
                ("Model family", model_name),
                ("Ansatz type", f"ADAPT-VQE (pool: {settings.get('adapt_pool', '?')})"),
                ("Drive enabled", False),
                ("L", settings.get("L")),
                ("Boundary", settings.get("boundary")),
                ("Ordering", settings.get("ordering")),
            ],
        ),
        (
            "Core physical parameters",
            [
                ("t", settings.get("t")),
                ("U", settings.get("u")),
                ("dv", settings.get("dv")),
            ],
        ),
        (
            "ADAPT controls",
            [
                ("ADAPT max depth", settings.get("adapt_max_depth", "?")),
                ("ADAPT eps_grad", settings.get("adapt_eps_grad", "?")),
                ("ADAPT eps_energy", settings.get("adapt_eps_energy", "?")),
                ("Inner optimizer", settings.get("adapt_inner_optimizer", "?")),
                ("Finite-angle fallback", settings.get("adapt_finite_angle_fallback", "?")),
                ("Finite-angle probe", settings.get("adapt_finite_angle", "?")),
            ],
        ),
        (
            "Trajectory settings",
            [
                ("trotter_steps", settings.get("trotter_steps")),
                ("t_final", settings.get("t_final")),
                ("Suzuki order", settings.get("suzuki_order")),
                ("Initial state source", settings.get("initial_state_source")),
            ],
        ),
    ]
    if problem == "hh":
        manifest_sections.append(
            (
                "Hubbard-Holstein parameters",
                [
                    ("omega0", settings.get("omega0")),
                    ("g_ep", settings.get("g_ep")),
                    ("n_ph_max", settings.get("n_ph_max")),
                    ("Boson encoding", settings.get("boson_encoding")),
                ],
            )
        )
    if str(settings.get("adapt_inner_optimizer", "")).strip().upper() == "SPSA":
        adapt_spsa = settings.get("adapt_spsa", {})
        if isinstance(adapt_spsa, dict):
            manifest_sections.append(
                (
                    "SPSA settings",
                    [
                        ("a", adapt_spsa.get("a")),
                        ("c", adapt_spsa.get("c")),
                        ("A", adapt_spsa.get("A")),
                        ("alpha", adapt_spsa.get("alpha")),
                        ("gamma", adapt_spsa.get("gamma")),
                        ("eval_repeats", adapt_spsa.get("eval_repeats")),
                        ("eval_agg", adapt_spsa.get("eval_agg")),
                        ("avg_last", adapt_spsa.get("avg_last")),
                    ],
                )
            )

    summary_sections: list[tuple[str, list[tuple[str, Any]]]] = [
        (
            "ADAPT outcome",
            [
                ("ADAPT-VQE energy", adapt.get("energy")),
                ("Exact GS energy", adapt.get("exact_gs_energy")),
                ("|ΔE|", adapt.get("abs_delta_e")),
                ("Ansatz depth", adapt.get("ansatz_depth")),
                ("Pool size", adapt.get("pool_size")),
            ],
        ),
        (
            "Optimization summary",
            [
                ("Stop reason", adapt.get("stop_reason")),
                ("Total nfev", adapt.get("nfev_total")),
                ("Elapsed (s)", adapt.get("elapsed_s")),
                ("Inner optimizer", settings.get("adapt_inner_optimizer")),
            ],
        ),
        (
            "Trajectory grid",
            [
                ("trotter_steps", settings.get("trotter_steps")),
                ("t_final", settings.get("t_final")),
                ("Initial state source", settings.get("initial_state_source")),
            ],
        ),
    ]

    operator_lines = [
        "Selected operators",
        "",
        f"Ansatz depth: {adapt.get('ansatz_depth')}",
        f"Pool size: {adapt.get('pool_size')}",
        f"Stop reason: {adapt.get('stop_reason')}",
        "",
    ]
    for op_label in (adapt.get("operators") or []):
        operator_lines.append(f"  {op_label}")

    with PdfPages(str(pdf_path)) as pdf:
        render_manifest_overview_page(
            pdf,
            title=f"{model_name} ADAPT-VQE report — L={settings.get('L')}",
            experiment_statement="ADAPT-VQE state preparation followed by exact-versus-Trotter trajectory diagnostics.",
            sections=manifest_sections,
            notes=[
                "The full operator list and executed command are moved to the appendix.",
            ],
        )
        render_executive_summary_page(
            pdf,
            title="Executive summary",
            experiment_statement="Prepared-state quality and convergence summary before trajectory pages.",
            sections=summary_sections,
            notes=[
                "Trajectory pages show fidelity, energy, occupations, and doublon from the ADAPT state.",
            ],
        )
        render_section_divider_page(
            pdf,
            title="Trajectory diagnostics",
            summary="Main result pages compare exact and Trotter trajectories starting from the ADAPT-prepared state.",
            bullets=[
                "Fidelity and energy.",
                "Site-0 occupations and doublon.",
            ],
        )

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

        render_section_divider_page(
            pdf,
            title="Technical appendix",
            summary="Detailed operator provenance and full reproducibility material.",
            bullets=[
                "Selected operator list.",
                "Executed command.",
            ],
        )
        render_text_page(pdf, operator_lines)
        render_command_page(
            pdf,
            run_command,
            script_name="pipelines/hardcoded/adapt_pipeline.py",
        )


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Hardcoded ADAPT-VQE Hubbard / Hubbard-Holstein pipeline. "
            "Direct HH CLI runs default to phase3_v1 continuation."
        )
    )
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--u", type=float, default=4.0)
    p.add_argument("--problem", choices=["hubbard", "hh"], default="hubbard")
    p.add_argument("--dv", type=float, default=0.0)
    p.add_argument("--omega0", type=float, default=0.0)
    p.add_argument("--g-ep", type=float, default=0.0, help="Holstein electron-phonon coupling g.")
    p.add_argument("--n-ph-max", type=int, default=1)
    p.add_argument("--boson-encoding", choices=["binary"], default="binary")
    p.add_argument("--boundary", choices=["periodic", "open"], default="open")
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
            "pareto_lean",
            "pareto_lean_l3",
            "pareto_lean_l2",
            "pareto_lean_gate_pruned",
            "uccsd_paop_lf_full",
            "uccsd_otimes_paop_lf_std",
            "uccsd_otimes_paop_lf2_std",
            "uccsd_otimes_paop_bond_disp_std",
            "uccsd_otimes_paop_lf_std_seq2p",
            "uccsd_otimes_paop_lf2_std_seq2p",
            "uccsd_otimes_paop_bond_disp_std_seq2p",
            "paop",
            "paop_min",
            "paop_std",
            "paop_full",
            "paop_lf",
            "paop_lf_std",
            "paop_lf2_std",
            "paop_lf3_std",
            "paop_lf4_std",
            "paop_lf_full",
            "paop_sq_std",
            "paop_sq_full",
            "paop_bond_disp_std",
            "paop_hop_sq_std",
            "paop_pair_sq_std",
            "vlf_only",
            "sq_only",
            "vlf_sq",
            "sq_dens_only",
            "vlf_sq_dens",
        ],
        default=None,
        help=(
            "ADAPT pool family. If omitted, runtime resolves problem-aware defaults: "
            "hubbard->uccsd; hh+phase3_v1 (direct canonical path)->paop_lf_std core + residual full_meta; "
            "hh+legacy->full_meta compatibility path; hh+phase1_v1/phase2_v1->older compatibility paths with the same narrow core + residual full_meta curriculum. "
            "HH also supports opt-in scaffold-derived presets pareto_lean, pareto_lean_l3, and pareto_lean_l2."
        ),
    )
    p.add_argument(
        "--adapt-pool-class-filter-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON keep-spec for filtering the HH full_meta pool by operator class. "
            "Only valid together with --problem hh --adapt-pool full_meta."
        ),
    )
    p.add_argument(
        "--adapt-continuation-mode",
        choices=["legacy", "phase1_v1", "phase2_v1", "phase3_v1"],
        default=None,
        help=(
            "Continuation mode for ADAPT. On the direct HH CLI, omitting this now resolves to phase3_v1, the canonical current path with full Phase 3 cheap/rerank scoring. "
            "legacy, phase1_v1, and phase2_v1 remain explicit historical/compatibility modes; non-HH direct omission keeps legacy compatibility behavior."
        ),
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
        choices=["COBYLA", "POWELL", "SPSA"],
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
    p.add_argument(
        "--adapt-beam-live-branches",
        type=int,
        default=1,
        help="Requested live frontier size for ADAPT beam mode (1 keeps exact single-branch semantics).",
    )
    p.add_argument(
        "--adapt-beam-children-per-parent",
        type=int,
        default=None,
        help="Optional per-parent child cap for ADAPT beam mode; omitted resolves from live-branch policy.",
    )
    p.add_argument(
        "--adapt-beam-terminated-keep",
        type=int,
        default=None,
        help="Optional terminal-pool cap for ADAPT beam mode; omitted resolves from live-branch policy.",
    )
    p.add_argument("--phase1-lambda-F", type=float, default=1.0)
    p.add_argument("--phase1-lambda-compile", type=float, default=0.05)
    p.add_argument("--phase1-lambda-measure", type=float, default=0.02)
    p.add_argument("--phase1-lambda-leak", type=float, default=0.0)
    p.add_argument("--phase1-score-z-alpha", type=float, default=0.0)
    p.add_argument("--phase1-depth-ref", type=float, default=1.0)
    p.add_argument("--phase1-group-ref", type=float, default=1.0)
    p.add_argument("--phase1-shot-ref", type=float, default=1.0)
    p.add_argument("--phase1-family-ref", type=float, default=1.0)
    p.add_argument("--phase1-compile-cx-proxy-weight", type=float, default=1.0)
    p.add_argument("--phase1-compile-sq-proxy-weight", type=float, default=0.5)
    p.add_argument("--phase1-compile-rotation-step-weight", type=float, default=1.0)
    p.add_argument("--phase1-compile-position-shift-weight", type=float, default=1.0)
    p.add_argument("--phase1-compile-refit-active-weight", type=float, default=1.0)
    p.add_argument("--phase1-measure-groups-weight", type=float, default=1.0)
    p.add_argument("--phase1-measure-shots-weight", type=float, default=1.0)
    p.add_argument("--phase1-measure-reuse-weight", type=float, default=1.0)
    p.add_argument("--phase1-opt-dim-cost-scale", type=float, default=1.0)
    p.add_argument("--phase1-family-repeat-cost-scale", type=float, default=1.0)
    p.add_argument(
        "--phase1-shortlist-size",
        type=int,
        default=64,
        help="Maximum candidate count admitted into phase-1 probing before phase-2 full scoring.",
    )
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
        "--phase2-shortlist-fraction",
        type=float,
        default=0.2,
        help="Fraction of phase-1 records admitted into phase-2 full scoring before shortlist-size capping.",
    )
    p.add_argument(
        "--phase2-shortlist-size",
        type=int,
        default=12,
        help="Maximum phase-2 shortlist size after cheap screening.",
    )
    p.add_argument(
        "--phase2-lambda-H",
        type=float,
        default=1e-6,
        help="Phase-2 reduced-window Hessian ridge regularization lambda_H used in the inherited-window solve.",
    )
    p.add_argument(
        "--phase2-rho",
        type=float,
        default=0.25,
        help="Phase-2 trust-region radius rho used in reduced-gain scoring.",
    )
    p.add_argument(
        "--phase2-gamma-N",
        type=float,
        default=1.0,
        help="Phase-2 novelty exponent gamma_N applied to the reduced novelty factor in the full_v2 score.",
    )
    p.add_argument(
        "--phase2-score-z-alpha",
        type=float,
        default=None,
        help="Optional Phase-2/3 confidence multiplier z_alpha. Defaults to --phase1-score-z-alpha when omitted.",
    )
    p.add_argument(
        "--phase2-lambda-F",
        type=float,
        default=None,
        help="Optional Phase-2 cheap-ratio metric scale lambda_F. Defaults to --phase1-lambda-F when omitted.",
    )
    p.add_argument("--phase2-depth-ref", type=float, default=1.0)
    p.add_argument("--phase2-group-ref", type=float, default=1.0)
    p.add_argument("--phase2-shot-ref", type=float, default=1.0)
    p.add_argument("--phase2-optdim-ref", type=float, default=1.0)
    p.add_argument("--phase2-reuse-ref", type=float, default=1.0)
    p.add_argument("--phase2-family-ref", type=float, default=1.0)
    p.add_argument("--phase2-novelty-eps", type=float, default=1e-6)
    p.add_argument("--phase2-cheap-score-eps", type=float, default=1e-12)
    p.add_argument("--phase2-metric-floor", type=float, default=1e-12)
    p.add_argument("--phase2-reduced-metric-collapse-rel-tol", type=float, default=1e-8)
    p.add_argument("--phase2-ridge-growth-factor", type=float, default=10.0)
    p.add_argument("--phase2-ridge-max-steps", type=int, default=12)
    p.add_argument("--phase2-leakage-cap", type=float, default=1e6)
    p.add_argument("--phase2-compile-cx-proxy-weight", type=float, default=1.0)
    p.add_argument("--phase2-compile-sq-proxy-weight", type=float, default=0.5)
    p.add_argument("--phase2-compile-rotation-step-weight", type=float, default=1.0)
    p.add_argument("--phase2-compile-position-shift-weight", type=float, default=1.0)
    p.add_argument("--phase2-compile-refit-active-weight", type=float, default=1.0)
    p.add_argument("--phase2-measure-groups-weight", type=float, default=1.0)
    p.add_argument("--phase2-measure-shots-weight", type=float, default=1.0)
    p.add_argument("--phase2-measure-reuse-weight", type=float, default=1.0)
    p.add_argument("--phase2-opt-dim-cost-scale", type=float, default=1.0)
    p.add_argument("--phase2-family-repeat-cost-scale", type=float, default=1.0)
    p.add_argument(
        "--phase2-w-depth",
        type=float,
        default=0.2,
        help="Phase-2 burden weight on normalized depth / compile cost in K_full.",
    )
    p.add_argument(
        "--phase2-w-group",
        type=float,
        default=0.15,
        help="Phase-2 burden weight on normalized new-group measurement cost in K_full.",
    )
    p.add_argument(
        "--phase2-w-shot",
        type=float,
        default=0.15,
        help="Phase-2 burden weight on normalized new-shot cost in K_full.",
    )
    p.add_argument(
        "--phase2-w-optdim",
        type=float,
        default=0.1,
        help="Phase-2 burden weight on normalized optimizer-dimension cost in K_full.",
    )
    p.add_argument(
        "--phase2-w-reuse",
        type=float,
        default=0.1,
        help="Phase-2 burden weight on normalized reuse penalty in K_full.",
    )
    p.add_argument(
        "--phase2-w-lifetime",
        type=float,
        default=0.05,
        help="Phase-2 burden weight on the remaining-horizon lifetime multiplier when lifetime cost mode is enabled.",
    )
    p.add_argument(
        "--phase2-eta-L",
        type=float,
        default=0.0,
        help="Phase-2 leakage penalty exponent eta_L multiplying exp(-eta_L * leakage_penalty) in the full_v2 score.",
    )
    p.add_argument(
        "--phase2-motif-bonus-weight",
        type=float,
        default=0.05,
        help="Phase-2 motif bonus weight beta_motif added on top of the reduced geometric score.",
    )
    p.add_argument(
        "--phase2-duplicate-penalty-weight",
        type=float,
        default=0.0,
        help="Phase-2 duplicate-direction penalty weight beta_dup subtracted from the augmented selector score.",
    )
    p.add_argument(
        "--phase2-frontier-ratio",
        type=float,
        default=0.9,
        help="Phase-2 shortlist frontier ratio used after cheap screening and before the full rerank.",
    )
    p.add_argument(
        "--phase3-frontier-ratio",
        type=float,
        default=0.9,
        help="Phase-3 shortlist frontier ratio used on the full-score rerank before any batch decision.",
    )
    p.add_argument(
        "--phase3-tie-beam-score-ratio",
        type=float,
        default=1.0,
        help="When < 1, any Phase-3 candidate within this ratio of the best full_v2_score joins a temporary score-band beam.",
    )
    p.add_argument(
        "--phase3-tie-beam-abs-tol",
        type=float,
        default=0.0,
        help="Absolute full_v2 score tolerance for the temporary Phase-3 tie-beam band.",
    )
    p.add_argument(
        "--phase3-tie-beam-max-branches",
        type=int,
        default=1,
        help="Maximum temporary branch count admitted from a Phase-3 score band. Values >1 activate conditional tie-beam branching.",
    )
    p.add_argument(
        "--phase3-tie-beam-max-late-coordinate",
        type=float,
        default=1.0,
        help="Disable Phase-3 tie-beam branching once the controller late-coordinate exceeds this value.",
    )
    p.add_argument(
        "--phase3-tie-beam-min-depth-left",
        type=int,
        default=0,
        help="Disable Phase-3 tie-beam branching once fewer than this many depths remain.",
    )
    p.set_defaults(phase2_enable_batching=True)
    p.add_argument("--phase2-enable-batching", dest="phase2_enable_batching", action="store_true")
    p.add_argument("--phase2-no-batching", dest="phase2_enable_batching", action="store_false")
    p.add_argument(
        "--phase2-batch-target-size",
        type=int,
        default=2,
        help="Target number of near-degenerate candidates admitted into a phase-2 batch selection step.",
    )
    p.add_argument(
        "--phase2-batch-size-cap",
        type=int,
        default=3,
        help="Hard cap on candidates admitted into a phase-2 batch selection step.",
    )
    p.add_argument(
        "--phase2-batch-near-degenerate-ratio",
        type=float,
        default=0.9,
        help="Relative full-score threshold eta_nd for the near-degenerate batch shell after shortlisting.",
    )
    p.add_argument(
        "--phase2-batch-rank-rel-tol",
        type=float,
        default=1e-6,
        help="Relative rank-floor tolerance tau_rank used by the reduced-plane batch admissibility test.",
    )
    p.add_argument(
        "--phase2-batch-additivity-tol",
        type=float,
        default=0.25,
        help="Additivity-defect tolerance tau_add used by the reduced-plane batch admissibility test.",
    )
    p.add_argument(
        "--phase2-compat-overlap-weight",
        type=float,
        default=0.4,
        help="Compatibility prescreen weight on support overlap in heuristic batch selection penalties.",
    )
    p.add_argument(
        "--phase2-compat-comm-weight",
        type=float,
        default=0.2,
        help="Compatibility prescreen weight on noncommutation in heuristic batch selection penalties.",
    )
    p.add_argument(
        "--phase2-compat-curv-weight",
        type=float,
        default=0.2,
        help="Compatibility prescreen weight on cross-curvature in heuristic batch selection penalties.",
    )
    p.add_argument(
        "--phase2-compat-sched-weight",
        type=float,
        default=0.2,
        help="Compatibility prescreen weight on schedule mismatch in heuristic batch selection penalties.",
    )
    p.add_argument(
        "--phase2-compat-measure-weight",
        type=float,
        default=0.2,
        help="Compatibility prescreen weight on measurement mismatch in heuristic batch selection penalties.",
    )
    p.add_argument(
        "--phase2-remaining-evaluations-proxy-mode",
        choices=["auto", "none", "remaining_depth"],
        default="auto",
        help=(
            "Remaining-horizon proxy mode used inside lifetime burden bookkeeping. "
            "'auto' preserves the previous behavior: remaining_depth when lifetime mode is on, otherwise none."
        ),
    )
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
        help=(
            "Phase-3 runtime split mode for the public HH ADAPT surface. "
            "The manuscript-facing canonical path keeps this fixed to 'off'; "
            "use 'shortlist_pauli_children_v1' to restore the archival child-set shortlist pathway."
        ),
    )
    p.add_argument(
        "--phase3-backend-cost-mode",
        choices=["proxy", "transpile_single_v1", "transpile_shortlist_v1"],
        default="proxy",
        help="Keep ADAPT logical but replace the Phase 3 compile-burden proxy with transpilation-derived burden against one backend or a fixed backend shortlist.",
    )
    p.add_argument(
        "--phase3-backend-name",
        type=str,
        default=None,
        help="Target backend name for --phase3-backend-cost-mode transpile_single_v1 (for example ibm_boston or ibm_miami).",
    )
    p.add_argument(
        "--phase3-backend-shortlist",
        type=str,
        default=None,
        help="Comma-separated backend shortlist for --phase3-backend-cost-mode transpile_shortlist_v1.",
    )
    p.add_argument(
        "--phase3-backend-transpile-seed",
        type=int,
        default=7,
        help="Seed used by the backend-conditioned transpilation oracle.",
    )
    p.add_argument(
        "--phase3-backend-optimization-level",
        type=int,
        default=1,
        help="Qiskit transpiler optimization level used by the backend-conditioned transpilation oracle.",
    )
    p.add_argument(
        "--phase3-selector-debug-topk",
        type=int,
        default=0,
        help="Emit compact phase3 selector top-k scoring logs per depth (0 disables).",
    )
    p.add_argument(
        "--phase3-selector-debug-max-depth",
        type=int,
        default=0,
        help="Maximum depth for selector debug logging (0 means all depths when enabled).",
    )
    p.add_argument(
        "--phase3-oracle-gradient-mode",
        choices=["off", "ideal", "shots", "aer_noise", "backend_scheduled", "runtime"],
        default="off",
        help=(
            "Opt-in direct HH phase3_v1 local oracle-gradient mode. Candidate gradient scouting uses expectation or raw-shot oracle finite-difference energies; inner re-optimization stays exact unless --phase3-oracle-inner-objective-mode noisy_v1 is selected."
        ),
    )
    p.add_argument(
        "--phase3-oracle-shots",
        type=int,
        default=2048,
        help="Shots per oracle circuit when --phase3-oracle-gradient-mode is enabled.",
    )
    p.add_argument(
        "--phase3-oracle-repeats",
        type=int,
        default=1,
        help="Repeat count for oracle gradient circuits; mean aggregate only in v1.",
    )
    p.add_argument(
        "--phase3-oracle-aggregate",
        choices=["mean"],
        default="mean",
        help="Aggregate for repeated oracle gradient circuits. v1 supports mean only.",
    )
    p.add_argument(
        "--phase3-oracle-backend-name",
        type=str,
        default=None,
        help="Backend name for aer_noise/backend_scheduled oracle modes (for example FakeNighthawk).",
    )
    p.add_argument(
        "--phase3-oracle-use-fake-backend",
        action="store_true",
        help="Use an offline fake backend for phase3 oracle-gradient mode.",
    )
    p.add_argument(
        "--phase3-oracle-seed",
        type=int,
        default=7,
        help="Seed for local oracle-gradient execution when enabled.",
    )
    p.add_argument(
        "--phase3-oracle-gradient-step",
        type=float,
        default=None,
        help="Finite-difference step for oracle-backed phase3 candidate gradients. Defaults to --adapt-finite-angle when omitted.",
    )
    p.add_argument(
        "--phase3-oracle-mitigation",
        choices=["none", "readout"],
        default="none",
        help=(
            "Base mitigation mode for phase3 oracle-gradient execution. "
            "Combine with --phase3-oracle-zne-scales, --phase3-oracle-local-gate-twirling, "
            "and --phase3-oracle-dd-sequence on the backend_scheduled path."
        ),
    )
    p.add_argument(
        "--phase3-oracle-local-readout-strategy",
        choices=["mthree"],
        default=None,
        help="Local readout mitigation strategy for phase3 oracle-gradient execution.",
    )
    p.add_argument(
        "--phase3-oracle-zne-scales",
        type=str,
        default=None,
        help="Comma-separated odd integer local ZNE scales for backend_scheduled phase3 oracle-gradient execution.",
    )
    p.add_argument(
        "--phase3-oracle-local-gate-twirling",
        action="store_true",
        help="Enable local two-qubit Pauli/gate twirling for backend_scheduled phase3 oracle-gradient execution.",
    )
    p.add_argument(
        "--phase3-oracle-dd-sequence",
        type=str,
        default=None,
        help="Enable local DD for backend_scheduled phase3 oracle-gradient execution (currently XpXm only).",
    )
    p.add_argument(
        "--phase3-oracle-execution-surface",
        choices=["auto", "expectation_v1", "raw_measurement_v1"],
        default="auto",
        help="Execution surface for phase3 oracle-gradient mode. 'auto' selects raw-shot only for runtime with mitigation=none.",
    )
    p.add_argument(
        "--phase3-oracle-inner-objective-mode",
        choices=["exact", "noisy_v1"],
        default="exact",
        help="When noisy_v1, HH phase3_v1 inner re-optimization uses the same oracle energy surface as candidate scouting (expectation_v1 or raw_measurement_v1). The runtime path reuses parameterized compiled templates so noisy SPSA stays enabled without recompiling each evaluation from scratch.",
    )
    p.add_argument(
        "--phase3-oracle-raw-transport",
        choices=["auto", "sampler_v2"],
        default="auto",
        help="Raw transport preference when phase3 oracle execution surface resolves to raw_measurement_v1 on the runtime sampler path.",
    )
    p.add_argument(
        "--phase3-oracle-raw-store-memory",
        action="store_true",
        help="Keep emitted raw measurement records in memory during phase3 raw-shot scouting.",
    )
    p.add_argument(
        "--phase3-oracle-raw-artifact-path",
        type=str,
        default=None,
        help="Optional NDJSON(.gz) path for phase3 raw-shot measurement records.",
    )
    p.add_argument(
        "--phase3-oracle-seed-transpiler",
        type=int,
        default=None,
        help="Optional transpiler seed for phase3 oracle execution.",
    )
    p.add_argument(
        "--phase3-oracle-transpile-optimization-level",
        type=int,
        default=1,
        help="Qiskit transpiler optimization level used by phase3 oracle execution.",
    )
    p.add_argument(
        "--final-noise-audit-mode",
        choices=["off", "ideal", "shots", "aer_noise", "backend_scheduled", "runtime"],
        default="off",
        help=(
            "Opt-in post-run final noise audit for the canonical direct HH ADAPT path. "
            "Current support is expectation-only, including runtime expectation audit; raw audit remains deferred."
        ),
    )
    p.add_argument(
        "--final-noise-audit-shots",
        type=int,
        default=2048,
        help="Shots per audit evaluation when --final-noise-audit-mode is enabled.",
    )
    p.add_argument(
        "--final-noise-audit-repeats",
        type=int,
        default=1,
        help="Repeat count for final noise audit evaluation; mean aggregate only in v1.",
    )
    p.add_argument(
        "--final-noise-audit-aggregate",
        choices=["mean"],
        default="mean",
        help="Aggregate for repeated final noise audit evaluations. v1 supports mean only.",
    )
    p.add_argument(
        "--final-noise-audit-backend-name",
        type=str,
        default=None,
        help="Backend name for final noise audit in backend_scheduled or runtime mode.",
    )
    p.add_argument(
        "--final-noise-audit-use-fake-backend",
        action="store_true",
        help="Use an offline fake backend for final noise audit when supported (backend_scheduled only).",
    )
    p.add_argument(
        "--final-noise-audit-seed",
        type=int,
        default=7,
        help="Seed for final noise audit execution.",
    )
    p.add_argument(
        "--final-noise-audit-mitigation",
        choices=["none", "readout"],
        default="none",
        help=(
            "Base mitigation mode for final noise audit. Use the backend_scheduled local knobs "
            "for ZNE/twirling/DD, or a named runtime profile on the runtime path."
        ),
    )
    p.add_argument(
        "--final-noise-audit-local-readout-strategy",
        choices=["mthree"],
        default=None,
        help="Local readout mitigation strategy for backend_scheduled final noise audit when readout mitigation is enabled.",
    )
    p.add_argument(
        "--final-noise-audit-zne-scales",
        type=str,
        default=None,
        help="Comma-separated odd integer local ZNE scales for backend_scheduled final noise audit.",
    )
    p.add_argument(
        "--final-noise-audit-local-gate-twirling",
        action="store_true",
        help="Enable local two-qubit Pauli/gate twirling for backend_scheduled final noise audit.",
    )
    p.add_argument(
        "--final-noise-audit-dd-sequence",
        type=str,
        default=None,
        help="Enable local DD for backend_scheduled final noise audit (currently XpXm only).",
    )
    p.add_argument(
        "--final-noise-audit-runtime-profile",
        choices=[
            "legacy_runtime_v0",
            "main_twirled_readout_v1",
            "dd_probe_twirled_readout_v1",
            "final_audit_zne_twirled_readout_v1",
        ],
        default="legacy_runtime_v0",
        help="Named runtime mitigation/suppression profile for final runtime expectation audit.",
    )
    p.add_argument(
        "--final-noise-audit-runtime-session-policy",
        choices=["prefer_session", "require_session", "backend_only"],
        default="prefer_session",
        help="Runtime session policy for final runtime expectation audit.",
    )
    p.add_argument(
        "--final-noise-audit-compare-unmitigated-baseline",
        action="store_true",
        help="Also evaluate an unmitigated baseline on the same final ADAPT state for comparison.",
    )
    p.add_argument(
        "--final-noise-audit-seed-transpiler",
        type=int,
        default=None,
        help="Optional transpiler seed for final noise audit execution.",
    )
    p.add_argument(
        "--final-noise-audit-transpile-optimization-level",
        type=int,
        default=1,
        help="Qiskit transpiler optimization level used by final noise audit execution.",
    )
    p.add_argument(
        "--final-noise-audit-strict",
        action="store_true",
        help="Fail the run if final noise audit initialization or execution fails.",
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
    p.add_argument(
        "--adapt-analytic-noise-std",
        type=float,
        default=0.0,
        help="Std-dev of run-local Gaussian noise injected into exact ADAPT search-time energy and exact commutator gradients (0 = disabled).",
    )
    p.add_argument(
        "--adapt-analytic-noise-seed",
        type=int,
        default=None,
        help="Optional RNG seed for run-local ADAPT analytic Gaussian noise draws.",
    )
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
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
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
    ansatz_input_state_for_adapt: np.ndarray | None = None
    ansatz_input_state_source = "hf"
    ansatz_input_state_kind: str | None = "reference_state"
    if args.adapt_ref_json is not None:
        psi_ref_override_for_adapt, adapt_ref_meta = _load_adapt_initial_state(
            Path(args.adapt_ref_json),
            int(nq_total),
        )
        ansatz_input_state_for_adapt = np.asarray(psi_ref_override_for_adapt, dtype=complex).reshape(-1)
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
            "initial_state_handoff_state_kind": adapt_ref_meta.get("initial_state_handoff_state_kind"),
            "settings": adapt_ref_meta.get("settings", {}),
            "adapt_vqe": adapt_ref_meta.get("adapt_vqe", {}),
            "adapt_ref_base_depth": int(adapt_ref_base_depth),
        }
        ansatz_input_state_source = str(adapt_ref_meta.get("initial_state_source") or "adapt_ref_json")
        raw_kind = adapt_ref_meta.get("initial_state_handoff_state_kind")
        ansatz_input_state_kind = None if raw_kind in {None, ""} else str(raw_kind)
        _ai_log(
            "hardcoded_adapt_ref_json_loaded",
            path=str(Path(args.adapt_ref_json)),
            initial_state_source=adapt_ref_meta.get("initial_state_source"),
            adapt_ref_base_depth=int(adapt_ref_base_depth),
        )
    else:
        ansatz_input_state_for_adapt, ansatz_input_state_source, ansatz_input_state_kind = _default_adapt_input_state(
            problem=str(problem_key),
            num_sites=int(args.L),
            ordering=str(args.ordering),
            n_ph_max=int(args.n_ph_max),
            boson_encoding=str(args.boson_encoding),
        )

    cli_adapt_continuation_mode = _resolve_cli_adapt_continuation_mode(
        problem=problem_key,
        requested_mode=args.adapt_continuation_mode,
    )
    phase3_oracle_gradient_config: Phase3OracleGradientConfig | None = None
    final_noise_audit_config: FinalNoiseAuditConfig | None = None
    phase3_oracle_gradient_mode_key = str(args.phase3_oracle_gradient_mode).strip().lower()
    if phase3_oracle_gradient_mode_key != "off":
        phase3_oracle_gradient_config = _resolve_phase3_oracle_gradient_config(
            Phase3OracleGradientConfig(
                noise_mode=str(phase3_oracle_gradient_mode_key),
                shots=int(args.phase3_oracle_shots),
                oracle_repeats=int(args.phase3_oracle_repeats),
                oracle_aggregate=str(args.phase3_oracle_aggregate),
                backend_name=(
                    None
                    if args.phase3_oracle_backend_name in {None, ""}
                    else str(args.phase3_oracle_backend_name)
                ),
                use_fake_backend=bool(args.phase3_oracle_use_fake_backend),
                seed=int(args.phase3_oracle_seed),
                gradient_step=(
                    float(args.phase3_oracle_gradient_step)
                    if args.phase3_oracle_gradient_step is not None
                    else float(args.adapt_finite_angle)
                ),
                mitigation_mode=str(args.phase3_oracle_mitigation),
                local_readout_strategy=(
                    None
                    if args.phase3_oracle_local_readout_strategy in {None, ""}
                    else str(args.phase3_oracle_local_readout_strategy)
                ),
                zne_scales=(
                    ()
                    if args.phase3_oracle_zne_scales in {None, ""}
                    else str(args.phase3_oracle_zne_scales)
                ),
                local_gate_twirling=bool(args.phase3_oracle_local_gate_twirling),
                dd_sequence=(
                    None
                    if args.phase3_oracle_dd_sequence in {None, ""}
                    else str(args.phase3_oracle_dd_sequence)
                ),
                execution_surface_requested=str(args.phase3_oracle_execution_surface),
                raw_transport=str(args.phase3_oracle_raw_transport),
                raw_store_memory=bool(args.phase3_oracle_raw_store_memory),
                raw_artifact_path=(
                    None
                    if args.phase3_oracle_raw_artifact_path in {None, ""}
                    else str(args.phase3_oracle_raw_artifact_path)
                ),
                seed_transpiler=(
                    None
                    if args.phase3_oracle_seed_transpiler is None
                    else int(args.phase3_oracle_seed_transpiler)
                ),
                transpile_optimization_level=int(args.phase3_oracle_transpile_optimization_level),
            )
        )
        _validate_phase3_oracle_gradient_config(
            config=phase3_oracle_gradient_config,
            problem=str(problem_key),
            continuation_mode=str(cli_adapt_continuation_mode),
        )
    final_noise_audit_mode_key = str(args.final_noise_audit_mode).strip().lower()
    if final_noise_audit_mode_key != "off":
        final_noise_audit_config = _resolve_final_noise_audit_config(
            FinalNoiseAuditConfig(
                noise_mode=str(final_noise_audit_mode_key),
                shots=int(args.final_noise_audit_shots),
                oracle_repeats=int(args.final_noise_audit_repeats),
                oracle_aggregate=str(args.final_noise_audit_aggregate),
                backend_name=(
                    None
                    if args.final_noise_audit_backend_name in {None, ""}
                    else str(args.final_noise_audit_backend_name)
                ),
                use_fake_backend=bool(args.final_noise_audit_use_fake_backend),
                seed=int(args.final_noise_audit_seed),
                mitigation_mode=str(args.final_noise_audit_mitigation),
                local_readout_strategy=(
                    None
                    if args.final_noise_audit_local_readout_strategy in {None, ""}
                    else str(args.final_noise_audit_local_readout_strategy)
                ),
                zne_scales=(
                    ()
                    if args.final_noise_audit_zne_scales in {None, ""}
                    else str(args.final_noise_audit_zne_scales)
                ),
                local_gate_twirling=bool(args.final_noise_audit_local_gate_twirling),
                dd_sequence=(
                    None
                    if args.final_noise_audit_dd_sequence in {None, ""}
                    else str(args.final_noise_audit_dd_sequence)
                ),
                runtime_profile_name=str(args.final_noise_audit_runtime_profile),
                runtime_session_policy=str(args.final_noise_audit_runtime_session_policy),
                compare_unmitigated_baseline=bool(
                    args.final_noise_audit_compare_unmitigated_baseline
                ),
                seed_transpiler=(
                    None
                    if args.final_noise_audit_seed_transpiler is None
                    else int(args.final_noise_audit_seed_transpiler)
                ),
                transpile_optimization_level=int(
                    args.final_noise_audit_transpile_optimization_level
                ),
                strict=bool(args.final_noise_audit_strict),
            )
        )
        _validate_final_noise_audit_config(
            config=final_noise_audit_config,
            problem=str(problem_key),
        )

    # Sector-filtered exact ground state: ADAPT-VQE preserves particle number,
    # so compare against the GS within the same (n_alpha, n_beta) sector.
    # For HH: use fermion-only sector filtering (phonon qubits free).
    num_particles_main = half_filled_num_particles(int(args.L))
    gs_energy_exact, gs_energy_source, exact_energy_reuse_mismatches = _resolve_exact_energy_override_from_adapt_ref(
        adapt_ref_meta=adapt_ref_meta,
        args=args,
        problem=problem_key,
        continuation_mode=str(cli_adapt_continuation_mode),
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
            and str(cli_adapt_continuation_mode).strip().lower() in _HH_STAGED_CONTINUATION_MODES
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
            adapt_analytic_noise_std=float(args.adapt_analytic_noise_std),
            adapt_analytic_noise_seed=(
                None
                if args.adapt_analytic_noise_seed is None
                else int(args.adapt_analytic_noise_seed)
            ),
            adapt_state_backend=str(args.adapt_state_backend),
            adapt_reopt_policy=str(args.adapt_reopt_policy),
            adapt_window_size=int(args.adapt_window_size),
            adapt_window_topk=int(args.adapt_window_topk),
            adapt_full_refit_every=int(args.adapt_full_refit_every),
            adapt_final_full_refit=bool(str(args.adapt_final_full_refit).strip().lower() == "true"),
            adapt_continuation_mode=str(cli_adapt_continuation_mode),
            adapt_beam_live_branches=int(args.adapt_beam_live_branches),
            adapt_beam_children_per_parent=(
                int(args.adapt_beam_children_per_parent)
                if args.adapt_beam_children_per_parent is not None
                else None
            ),
            adapt_beam_terminated_keep=(
                int(args.adapt_beam_terminated_keep)
                if args.adapt_beam_terminated_keep is not None
                else None
            ),
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
            phase1_depth_ref=float(args.phase1_depth_ref),
            phase1_group_ref=float(args.phase1_group_ref),
            phase1_shot_ref=float(args.phase1_shot_ref),
            phase1_family_ref=float(args.phase1_family_ref),
            phase1_compile_cx_proxy_weight=float(args.phase1_compile_cx_proxy_weight),
            phase1_compile_sq_proxy_weight=float(args.phase1_compile_sq_proxy_weight),
            phase1_compile_rotation_step_weight=float(args.phase1_compile_rotation_step_weight),
            phase1_compile_position_shift_weight=float(args.phase1_compile_position_shift_weight),
            phase1_compile_refit_active_weight=float(args.phase1_compile_refit_active_weight),
            phase1_measure_groups_weight=float(args.phase1_measure_groups_weight),
            phase1_measure_shots_weight=float(args.phase1_measure_shots_weight),
            phase1_measure_reuse_weight=float(args.phase1_measure_reuse_weight),
            phase1_opt_dim_cost_scale=float(args.phase1_opt_dim_cost_scale),
            phase1_family_repeat_cost_scale=float(args.phase1_family_repeat_cost_scale),
            phase1_shortlist_size=int(args.phase1_shortlist_size),
            phase1_probe_max_positions=int(args.phase1_probe_max_positions),
            phase1_plateau_patience=int(args.phase1_plateau_patience),
            phase1_trough_margin_ratio=float(args.phase1_trough_margin_ratio),
            phase1_prune_enabled=bool(args.phase1_prune_enabled),
            phase1_prune_fraction=float(args.phase1_prune_fraction),
            phase1_prune_max_candidates=int(args.phase1_prune_max_candidates),
            phase1_prune_max_regression=float(args.phase1_prune_max_regression),
            phase2_shortlist_fraction=float(args.phase2_shortlist_fraction),
            phase2_shortlist_size=int(args.phase2_shortlist_size),
            phase2_lambda_H=float(args.phase2_lambda_H),
            phase2_rho=float(args.phase2_rho),
            phase2_gamma_N=float(args.phase2_gamma_N),
            phase2_score_z_alpha=(
                float(args.phase2_score_z_alpha)
                if args.phase2_score_z_alpha is not None
                else None
            ),
            phase2_lambda_F=(
                float(args.phase2_lambda_F)
                if args.phase2_lambda_F is not None
                else None
            ),
            phase2_depth_ref=float(args.phase2_depth_ref),
            phase2_group_ref=float(args.phase2_group_ref),
            phase2_shot_ref=float(args.phase2_shot_ref),
            phase2_optdim_ref=float(args.phase2_optdim_ref),
            phase2_reuse_ref=float(args.phase2_reuse_ref),
            phase2_family_ref=float(args.phase2_family_ref),
            phase2_novelty_eps=float(args.phase2_novelty_eps),
            phase2_cheap_score_eps=float(args.phase2_cheap_score_eps),
            phase2_metric_floor=float(args.phase2_metric_floor),
            phase2_reduced_metric_collapse_rel_tol=float(
                args.phase2_reduced_metric_collapse_rel_tol
            ),
            phase2_ridge_growth_factor=float(args.phase2_ridge_growth_factor),
            phase2_ridge_max_steps=int(args.phase2_ridge_max_steps),
            phase2_leakage_cap=float(args.phase2_leakage_cap),
            phase2_compile_cx_proxy_weight=float(args.phase2_compile_cx_proxy_weight),
            phase2_compile_sq_proxy_weight=float(args.phase2_compile_sq_proxy_weight),
            phase2_compile_rotation_step_weight=float(args.phase2_compile_rotation_step_weight),
            phase2_compile_position_shift_weight=float(args.phase2_compile_position_shift_weight),
            phase2_compile_refit_active_weight=float(args.phase2_compile_refit_active_weight),
            phase2_measure_groups_weight=float(args.phase2_measure_groups_weight),
            phase2_measure_shots_weight=float(args.phase2_measure_shots_weight),
            phase2_measure_reuse_weight=float(args.phase2_measure_reuse_weight),
            phase2_opt_dim_cost_scale=float(args.phase2_opt_dim_cost_scale),
            phase2_family_repeat_cost_scale=float(args.phase2_family_repeat_cost_scale),
            phase2_w_depth=float(args.phase2_w_depth),
            phase2_w_group=float(args.phase2_w_group),
            phase2_w_shot=float(args.phase2_w_shot),
            phase2_w_optdim=float(args.phase2_w_optdim),
            phase2_w_reuse=float(args.phase2_w_reuse),
            phase2_w_lifetime=float(args.phase2_w_lifetime),
            phase2_eta_L=float(args.phase2_eta_L),
            phase2_motif_bonus_weight=float(args.phase2_motif_bonus_weight),
            phase2_duplicate_penalty_weight=float(args.phase2_duplicate_penalty_weight),
            phase2_frontier_ratio=float(args.phase2_frontier_ratio),
            phase3_frontier_ratio=float(args.phase3_frontier_ratio),
            phase3_tie_beam_score_ratio=float(args.phase3_tie_beam_score_ratio),
            phase3_tie_beam_abs_tol=float(args.phase3_tie_beam_abs_tol),
            phase3_tie_beam_max_branches=int(args.phase3_tie_beam_max_branches),
            phase3_tie_beam_max_late_coordinate=float(args.phase3_tie_beam_max_late_coordinate),
            phase3_tie_beam_min_depth_left=int(args.phase3_tie_beam_min_depth_left),
            phase2_enable_batching=bool(args.phase2_enable_batching),
            phase2_batch_target_size=int(args.phase2_batch_target_size),
            phase2_batch_size_cap=int(args.phase2_batch_size_cap),
            phase2_batch_near_degenerate_ratio=float(args.phase2_batch_near_degenerate_ratio),
            phase2_batch_rank_rel_tol=float(args.phase2_batch_rank_rel_tol),
            phase2_batch_additivity_tol=float(args.phase2_batch_additivity_tol),
            phase2_compat_overlap_weight=float(args.phase2_compat_overlap_weight),
            phase2_compat_comm_weight=float(args.phase2_compat_comm_weight),
            phase2_compat_curv_weight=float(args.phase2_compat_curv_weight),
            phase2_compat_sched_weight=float(args.phase2_compat_sched_weight),
            phase2_compat_measure_weight=float(args.phase2_compat_measure_weight),
            phase2_remaining_evaluations_proxy_mode=str(
                args.phase2_remaining_evaluations_proxy_mode
            ),
            adapt_pool_class_filter_json=(
                Path(args.adapt_pool_class_filter_json)
                if args.adapt_pool_class_filter_json is not None
                else None
            ),
            phase3_motif_source_json=(Path(args.phase3_motif_source_json) if args.phase3_motif_source_json is not None else None),
            phase3_symmetry_mitigation_mode=str(args.phase3_symmetry_mitigation_mode),
            phase3_enable_rescue=bool(args.phase3_enable_rescue),
            phase3_lifetime_cost_mode=str(args.phase3_lifetime_cost_mode),
            phase3_runtime_split_mode=str(args.phase3_runtime_split_mode),
            phase3_backend_cost_mode=str(args.phase3_backend_cost_mode),
            phase3_backend_name=(None if args.phase3_backend_name in {None, ""} else str(args.phase3_backend_name)),
            phase3_backend_shortlist=(
                []
                if args.phase3_backend_shortlist in {None, ""}
                else [str(tok).strip() for tok in str(args.phase3_backend_shortlist).split(",") if str(tok).strip() != ""]
            ),
            phase3_backend_transpile_seed=int(args.phase3_backend_transpile_seed),
            phase3_backend_optimization_level=int(args.phase3_backend_optimization_level),
            phase3_selector_debug_topk=int(args.phase3_selector_debug_topk),
            phase3_selector_debug_max_depth=int(args.phase3_selector_debug_max_depth),
            phase3_oracle_gradient_config=phase3_oracle_gradient_config,
            final_noise_audit_config=final_noise_audit_config,
            phase3_oracle_inner_objective_mode=str(args.phase3_oracle_inner_objective_mode),
        )
    except Exception as exc:
        _ai_log("hardcoded_adapt_vqe_failed", L=int(args.L), error=str(exc))
        adapt_payload = {
            "success": False,
            "method": f"hardcoded_adapt_vqe_{str(args.adapt_pool).lower()}",
            "energy": None,
            "adapt_inner_optimizer": str(args.adapt_inner_optimizer),
            "analytic_noise_applied": bool(float(args.adapt_analytic_noise_std) > 0.0),
            "analytic_noise_std": float(args.adapt_analytic_noise_std),
            "analytic_noise_seed": (
                None
                if args.adapt_analytic_noise_seed is None
                else int(args.adapt_analytic_noise_seed)
            ),
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
    initial_state_source_resolved = str(
        args.initial_state_source if args.initial_state_source != "adapt_vqe" or adapt_payload.get("success") else "exact"
    )
    initial_state_kind_resolved = (
        "prepared_state"
        if (args.initial_state_source == "adapt_vqe" and bool(adapt_payload.get("success", False)))
        else "reference_state"
    )

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
            "adapt_pool_class_filter_json": (
                str(args.adapt_pool_class_filter_json)
                if args.adapt_pool_class_filter_json is not None
                else None
            ),
            "adapt_pool_class_filter_classifier_version": (
                adapt_payload.get("adapt_pool_class_filter_classifier_version")
            ),
            "adapt_pool_class_filter_keep_classes": (
                adapt_payload.get("adapt_pool_class_filter_keep_classes")
            ),
            "adapt_continuation_mode": str(cli_adapt_continuation_mode),
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
            "adapt_analytic_noise_std": float(args.adapt_analytic_noise_std),
            "adapt_analytic_noise_seed": (
                None
                if args.adapt_analytic_noise_seed is None
                else int(args.adapt_analytic_noise_seed)
            ),
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
            "phase1_depth_ref": float(args.phase1_depth_ref),
            "phase1_group_ref": float(args.phase1_group_ref),
            "phase1_shot_ref": float(args.phase1_shot_ref),
            "phase1_family_ref": float(args.phase1_family_ref),
            "phase1_compile_cx_proxy_weight": float(args.phase1_compile_cx_proxy_weight),
            "phase1_compile_sq_proxy_weight": float(args.phase1_compile_sq_proxy_weight),
            "phase1_compile_rotation_step_weight": float(args.phase1_compile_rotation_step_weight),
            "phase1_compile_position_shift_weight": float(args.phase1_compile_position_shift_weight),
            "phase1_compile_refit_active_weight": float(args.phase1_compile_refit_active_weight),
            "phase1_measure_groups_weight": float(args.phase1_measure_groups_weight),
            "phase1_measure_shots_weight": float(args.phase1_measure_shots_weight),
            "phase1_measure_reuse_weight": float(args.phase1_measure_reuse_weight),
            "phase1_opt_dim_cost_scale": float(args.phase1_opt_dim_cost_scale),
            "phase1_family_repeat_cost_scale": float(args.phase1_family_repeat_cost_scale),
            "phase1_shortlist_size": int(args.phase1_shortlist_size),
            "phase1_probe_max_positions": int(args.phase1_probe_max_positions),
            "phase1_plateau_patience": int(args.phase1_plateau_patience),
            "phase1_trough_margin_ratio": float(args.phase1_trough_margin_ratio),
            "phase1_prune_enabled": bool(args.phase1_prune_enabled),
            "phase1_prune_fraction": float(args.phase1_prune_fraction),
            "phase1_prune_max_candidates": int(args.phase1_prune_max_candidates),
            "phase1_prune_max_regression": float(args.phase1_prune_max_regression),
            "phase2_shortlist_fraction": float(args.phase2_shortlist_fraction),
            "phase2_shortlist_size": int(args.phase2_shortlist_size),
            "phase2_lambda_H": float(args.phase2_lambda_H),
            "phase2_rho": float(args.phase2_rho),
            "phase2_gamma_N": float(args.phase2_gamma_N),
            "phase2_score_z_alpha": (
                float(args.phase2_score_z_alpha)
                if args.phase2_score_z_alpha is not None
                else None
            ),
            "phase2_lambda_F": (
                float(args.phase2_lambda_F)
                if args.phase2_lambda_F is not None
                else None
            ),
            "phase2_depth_ref": float(args.phase2_depth_ref),
            "phase2_group_ref": float(args.phase2_group_ref),
            "phase2_shot_ref": float(args.phase2_shot_ref),
            "phase2_optdim_ref": float(args.phase2_optdim_ref),
            "phase2_reuse_ref": float(args.phase2_reuse_ref),
            "phase2_family_ref": float(args.phase2_family_ref),
            "phase2_novelty_eps": float(args.phase2_novelty_eps),
            "phase2_cheap_score_eps": float(args.phase2_cheap_score_eps),
            "phase2_metric_floor": float(args.phase2_metric_floor),
            "phase2_reduced_metric_collapse_rel_tol": float(
                args.phase2_reduced_metric_collapse_rel_tol
            ),
            "phase2_ridge_growth_factor": float(args.phase2_ridge_growth_factor),
            "phase2_ridge_max_steps": int(args.phase2_ridge_max_steps),
            "phase2_leakage_cap": float(args.phase2_leakage_cap),
            "phase2_compile_cx_proxy_weight": float(args.phase2_compile_cx_proxy_weight),
            "phase2_compile_sq_proxy_weight": float(args.phase2_compile_sq_proxy_weight),
            "phase2_compile_rotation_step_weight": float(args.phase2_compile_rotation_step_weight),
            "phase2_compile_position_shift_weight": float(args.phase2_compile_position_shift_weight),
            "phase2_compile_refit_active_weight": float(args.phase2_compile_refit_active_weight),
            "phase2_measure_groups_weight": float(args.phase2_measure_groups_weight),
            "phase2_measure_shots_weight": float(args.phase2_measure_shots_weight),
            "phase2_measure_reuse_weight": float(args.phase2_measure_reuse_weight),
            "phase2_opt_dim_cost_scale": float(args.phase2_opt_dim_cost_scale),
            "phase2_family_repeat_cost_scale": float(args.phase2_family_repeat_cost_scale),
            "phase2_w_depth": float(args.phase2_w_depth),
            "phase2_w_group": float(args.phase2_w_group),
            "phase2_w_shot": float(args.phase2_w_shot),
            "phase2_w_optdim": float(args.phase2_w_optdim),
            "phase2_w_reuse": float(args.phase2_w_reuse),
            "phase2_w_lifetime": float(args.phase2_w_lifetime),
            "phase2_eta_L": float(args.phase2_eta_L),
            "phase2_motif_bonus_weight": float(args.phase2_motif_bonus_weight),
            "phase2_duplicate_penalty_weight": float(args.phase2_duplicate_penalty_weight),
            "phase2_frontier_ratio": float(args.phase2_frontier_ratio),
            "phase3_frontier_ratio": float(args.phase3_frontier_ratio),
            "phase3_tie_beam_score_ratio": float(args.phase3_tie_beam_score_ratio),
            "phase3_tie_beam_abs_tol": float(args.phase3_tie_beam_abs_tol),
            "phase3_tie_beam_max_branches": int(args.phase3_tie_beam_max_branches),
            "phase3_tie_beam_max_late_coordinate": float(args.phase3_tie_beam_max_late_coordinate),
            "phase3_tie_beam_min_depth_left": int(args.phase3_tie_beam_min_depth_left),
            "phase2_enable_batching": bool(args.phase2_enable_batching),
            "phase2_batch_target_size": int(args.phase2_batch_target_size),
            "phase2_batch_size_cap": int(args.phase2_batch_size_cap),
            "phase2_batch_near_degenerate_ratio": float(args.phase2_batch_near_degenerate_ratio),
            "phase2_batch_rank_rel_tol": float(args.phase2_batch_rank_rel_tol),
            "phase2_batch_additivity_tol": float(args.phase2_batch_additivity_tol),
            "phase2_compat_overlap_weight": float(args.phase2_compat_overlap_weight),
            "phase2_compat_comm_weight": float(args.phase2_compat_comm_weight),
            "phase2_compat_curv_weight": float(args.phase2_compat_curv_weight),
            "phase2_compat_sched_weight": float(args.phase2_compat_sched_weight),
            "phase2_compat_measure_weight": float(args.phase2_compat_measure_weight),
            "phase2_remaining_evaluations_proxy_mode": str(
                args.phase2_remaining_evaluations_proxy_mode
            ),
            "phase3_motif_source_json": (
                str(args.phase3_motif_source_json)
                if args.phase3_motif_source_json is not None
                else None
            ),
            "phase3_symmetry_mitigation_mode": str(args.phase3_symmetry_mitigation_mode),
            "phase3_enable_rescue": bool(args.phase3_enable_rescue),
            "phase3_lifetime_cost_mode": str(args.phase3_lifetime_cost_mode),
            "phase3_runtime_split_mode": str(args.phase3_runtime_split_mode),
            "phase3_backend_cost_mode": str(args.phase3_backend_cost_mode),
            "phase3_backend_name": (
                None if args.phase3_backend_name in {None, ""} else str(args.phase3_backend_name)
            ),
            "phase3_backend_shortlist": (
                []
                if args.phase3_backend_shortlist in {None, ""}
                else [str(tok).strip() for tok in str(args.phase3_backend_shortlist).split(",") if str(tok).strip() != ""]
            ),
            "phase3_backend_transpile_seed": int(args.phase3_backend_transpile_seed),
            "phase3_backend_optimization_level": int(args.phase3_backend_optimization_level),
            "phase3_oracle_inner_objective_mode": str(
                adapt_payload.get(
                    "phase3_oracle_inner_objective_mode",
                    args.phase3_oracle_inner_objective_mode,
                )
            ),
            "phase3_oracle_inner_objective_mode_requested": str(
                adapt_payload.get(
                    "phase3_oracle_inner_objective_mode_requested",
                    args.phase3_oracle_inner_objective_mode,
                )
            ),
            "phase3_oracle_inner_objective_runtime_guard_reason": (
                adapt_payload.get("phase3_oracle_inner_objective_runtime_guard_reason")
            ),
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
        "initial_state": build_statevector_manifest(
            psi_state=np.asarray(psi0, dtype=complex).reshape(-1),
            source=initial_state_source_resolved,
            handoff_state_kind=initial_state_kind_resolved,
            amplitude_cutoff=1e-12,
        ),
        "ansatz_input_state": build_statevector_manifest(
            psi_state=np.asarray(ansatz_input_state_for_adapt, dtype=complex).reshape(-1),
            source=str(ansatz_input_state_source),
            handoff_state_kind=ansatz_input_state_kind,
            amplitude_cutoff=1e-12,
        ),
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
        adapt_ref_import["ansatz_input_state_persisted"] = True
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
    _safe_stdout_print(f"Wrote JSON: {output_json}")
    if not args.skip_pdf:
        _safe_stdout_print(f"Wrote PDF:  {output_pdf}")


if __name__ == "__main__":
    main()
