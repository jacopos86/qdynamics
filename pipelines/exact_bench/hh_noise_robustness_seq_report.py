#!/usr/bin/env python3
"""HH noise-robustness sequential report (stage-transition + Magnus/Trotter).

Pipeline sequence:
1) HVA warm-start VQE
2) ADAPT-VQE with Pool B strict union: UCCSD_lifted + HVA + PAOP_FULL
3) Conventional VQE seeded from ADAPT final state

Then audits noiseless dynamics (Suzuki-2, midpoint Magnus-2, CFQM4/CFQM6, exact)
and noisy dynamics for selected methods (default: cfqm4,suzuki2) under
ideal/shots/aer_noise for static + drive profiles.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import queue as pyqueue
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter

from docs.reports.pdf_utils import (
    HAS_MATPLOTLIB,
    current_command_string,
    get_PdfPages,
    get_plt,
    render_command_page,
    render_compact_table,
    render_text_page,
    require_matplotlib,
)
from docs.reports.report_labels import (
    report_method_label,
    report_metric_label,
    report_stage_label,
)
from docs.reports.report_pages import (
    render_executive_summary_page,
    render_manifest_overview_page,
    render_section_divider_page,
)
from src.quantum.drives_time_potential import (
    build_gaussian_sinusoid_density_drive,
    evaluate_drive_waveform,
    reference_method_name,
)
from src.quantum.hartree_fock_reference_state import hubbard_holstein_reference_state
from src.quantum.hubbard_latex_python_pairs import (
    SPIN_DN,
    SPIN_UP,
    boson_qubits_per_site,
    build_hubbard_holstein_hamiltonian,
    mode_index,
)
from src.quantum.operator_pools import make_pool as make_paop_pool
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.spsa_optimizer import spsa_minimize
from src.quantum.time_propagation.cfqm_schemes import get_cfqm_scheme
from src.quantum.ansatz_parameterization import project_runtime_theta_block_mean
from src.quantum.vqe_latex_python_pairs import (
    AnsatzTerm,
    HardcodedUCCSDAnsatz,
    HubbardHolsteinPhysicalTermwiseAnsatz,
    exact_ground_energy_sector_hh,
    hamiltonian_matrix,
    vqe_minimize,
)

from pipelines.exact_bench.hh_seq_transition_utils import (
    TransitionConfig,
    TransitionState,
    build_pool_b_strict_union,
    build_time_dependent_sparse_qop,
    flatten_coeff_map_real_imag,
    summarize_transition,
    update_transition_state,
)
from pipelines.exact_bench.statevector_kernels import (
    apply_h_poly as _apply_pauli_polynomial,
    commutator_gradient as _commutator_gradient,
    compile_ansatz_executor as _compile_ansatz_executor_shared,
    compile_h_poly as _compile_h_poly_shared,
    energy_one_apply as _energy_one_apply_shared,
    prepare_state_for_ansatz as _prepare_state_for_ansatz,
)
from pipelines.exact_bench.noise_oracle_runtime import (
    BACKEND_SCHEDULED_ATTRIBUTION_SLICES,
    ExpectationOracle,
    OracleConfig,
    RawMeasurementOracle,
    _all_z_full_register_qop,
    _append_reference_state,
    _bootstrap_hh_full_register_z_artifact,
    _doublon_site_qop,
    _number_operator_qop,
    _summarize_hh_exact_diagonal_reference,
    _summarize_hh_full_register_z_records,
    _summarize_hh_full_register_z_records_postprocessed,
    normalize_ideal_reference_symmetry_mitigation,
    normalize_mitigation_config,
    normalize_runtime_estimator_profile_config,
    normalize_runtime_session_policy_config,
    normalize_sampler_raw_runtime_config,
    normalize_symmetry_mitigation_config,
)
from pipelines.hardcoded.adapt_circuit_cost import (
    _build_ansatz_circuit,
    _load_adapt_result,
    reconstruct_imported_adapt_circuit,
)
from pipelines.hardcoded.adapt_circuit_execution import (
    bind_parameterized_ansatz_circuit,
    build_parameterized_ansatz_plan,
)
from pipelines.hardcoded import hubbard_pipeline as hc_pipeline


def _ai_log(event: str, **fields: Any) -> None:
    payload = {
        "event": str(event),
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    print(f"AI_LOG {json.dumps(payload, sort_keys=True, default=str)}", flush=True)


_HH_MINIMUMS: dict[tuple[int, int], dict[str, Any]] = {
    (2, 1): {"trotter_steps": 64, "reps": 2, "restarts": 3, "maxiter": 800, "method": "COBYLA"},
    (2, 2): {"trotter_steps": 128, "reps": 3, "restarts": 4, "maxiter": 1500, "method": "COBYLA"},
    (3, 1): {"trotter_steps": 192, "reps": 2, "restarts": 4, "maxiter": 2400, "method": "COBYLA"},
}

_NOISY_METHODS_ALLOWED = {"suzuki2", "cfqm4", "cfqm6"}


def _half_filled_particles(num_sites: int) -> tuple[int, int]:
    return ((int(num_sites) + 1) // 2, int(num_sites) // 2)


def _collect_hardcoded_terms_exyz(poly: Any, tol: float = 1e-12) -> tuple[list[str], dict[str, complex]]:
    coeff_map: dict[str, complex] = {}
    order: list[str] = []
    for term in poly.return_polynomial():
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tol):
            continue
        if label not in coeff_map:
            coeff_map[label] = 0.0 + 0.0j
            order.append(label)
        coeff_map[label] += coeff
    cleaned_order = [lbl for lbl in order if abs(coeff_map[lbl]) > float(tol)]
    cleaned_map = {lbl: coeff_map[lbl] for lbl in cleaned_order}
    return cleaned_order, cleaned_map


def _parse_noisy_methods_csv(raw: str) -> list[str]:
    vals: list[str] = []
    for tok in str(raw).split(","):
        t = str(tok).strip().lower()
        if not t:
            continue
        if t not in _NOISY_METHODS_ALLOWED:
            raise ValueError(
                f"Unsupported noisy method {t!r}. Allowed: {sorted(_NOISY_METHODS_ALLOWED)}"
            )
        if t not in vals:
            vals.append(t)
    if not vals:
        raise ValueError("Expected at least one noisy method in --noisy-methods.")
    return vals


def _normalize_display_string_list(raw: Any, *, default: list[str]) -> list[str]:
    values: list[str] = []
    if isinstance(raw, (list, tuple, set)):
        values = [str(item).strip() for item in raw if str(item).strip()]
    elif raw is not None and str(raw).strip():
        values = [tok.strip() for tok in str(raw).split(",") if tok.strip()]
    return values or [str(item).strip() for item in default if str(item).strip()]


def _build_mitigation_config(
    *,
    mitigation: str,
    zne_scales: str | None,
    dd_sequence: str | None,
) -> dict[str, Any]:
    return normalize_mitigation_config(
        {
            "mode": str(mitigation),
            "zne_scales": zne_scales,
            "dd_sequence": dd_sequence,
        }
    )


def _build_symmetry_mitigation_config(
    *,
    mode: str,
    L: int,
    ordering: str,
) -> dict[str, Any]:
    num_particles = _half_filled_particles(int(L))
    return normalize_symmetry_mitigation_config(
        {
            "mode": str(mode),
            "num_sites": int(L),
            "ordering": str(ordering),
            "sector_n_up": int(num_particles[0]),
            "sector_n_dn": int(num_particles[1]),
        }
    )


def _normalize_state(psi: np.ndarray) -> np.ndarray:
    arr = np.asarray(psi, dtype=complex).reshape(-1)
    nrm = float(np.linalg.norm(arr))
    if nrm <= 0.0:
        raise ValueError("Encountered zero norm state.")
    return arr / nrm


def _build_hh_hamiltonian(args: argparse.Namespace) -> Any:
    return build_hubbard_holstein_hamiltonian(
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
        pbc=(str(args.boundary).strip().lower() == "periodic"),
        include_zero_point=True,
    )


def _import_amplitudes_qn_to_q0_to_statevector(
    amps_payload: dict[str, Any],
    *,
    nq: int,
) -> np.ndarray:
    state = np.zeros(int(1 << int(nq)), dtype=complex)
    for bitstr_raw, coeff_raw in amps_payload.items():
        coeff_map = dict(coeff_raw) if isinstance(coeff_raw, dict) else {}
        amp = complex(float(coeff_map.get("re", 0.0)), float(coeff_map.get("im", 0.0)))
        state[int(str(bitstr_raw), 2)] = amp
    return _normalize_state(state)


def _classify_imported_payload(raw_payload: dict[str, Any]) -> str:
    if not isinstance(raw_payload, dict):
        return "unknown"
    adapt_vqe = raw_payload.get("adapt_vqe", {})
    if isinstance(raw_payload.get("ground_state"), dict):
        return "direct_adapt_artifact"
    if isinstance(adapt_vqe, dict) and isinstance(adapt_vqe.get("continuation"), dict):
        return "direct_adapt_artifact"
    if isinstance(raw_payload.get("continuation"), dict) or isinstance(raw_payload.get("meta"), dict):
        return "handoff_bundle"
    return "unknown"


def _load_imported_artifact_context(artifact_json: str | Path) -> dict[str, Any]:
    path = Path(artifact_json)
    raw_payload = _load_adapt_result(path)
    bundle = reconstruct_imported_adapt_circuit(raw_payload)
    payload = bundle["payload"]
    settings = dict(bundle["settings"])
    ordered_labels_exyz, static_coeff_map_exyz = _collect_hardcoded_terms_exyz(bundle["h_poly"])
    prepared_state = None
    init = payload.get("initial_state", {}) if isinstance(payload, dict) else {}
    if isinstance(init, dict) and isinstance(init.get("amplitudes_qn_to_q0"), dict):
        nq = int(init.get("nq_total", bundle["num_qubits"]))
        prepared_state = _import_amplitudes_qn_to_q0_to_statevector(
            dict(init["amplitudes_qn_to_q0"]),
            nq=int(nq),
        )
    exact_energy = None
    adapt_vqe = payload.get("adapt_vqe", {}) if isinstance(payload, dict) else {}
    if isinstance(adapt_vqe, dict):
        exact_energy = adapt_vqe.get("exact_gs_energy", None)
    if exact_energy is None and isinstance(payload.get("exact", {}), dict):
        exact_energy = payload["exact"].get("E_exact_sector", None)
    return {
        "path": path,
        "source_kind": _classify_imported_payload(dict(raw_payload) if isinstance(raw_payload, dict) else {}),
        "payload": payload,
        "settings": settings,
        "ordered_labels_exyz": list(ordered_labels_exyz),
        "static_coeff_map_exyz": dict(static_coeff_map_exyz),
        "prepared_state": prepared_state,
        "ansatz_input_state": bundle.get("ansatz_input_state", None),
        "ansatz_input_state_meta": dict(bundle.get("ansatz_input_state_meta", {})),
        "circuit": bundle["circuit"],
        "layout": bundle["layout"],
        "theta_runtime": bundle["theta_runtime"],
        "num_qubits": int(bundle["num_qubits"]),
        "saved_energy": (
            None if not isinstance(adapt_vqe, dict) else adapt_vqe.get("energy", None)
        ),
        "exact_energy": exact_energy,
    }


def _imported_pool_type(ctx: Mapping[str, Any]) -> str | None:
    payload = ctx.get("payload", {}) if isinstance(ctx, Mapping) else {}
    adapt_vqe = payload.get("adapt_vqe", {}) if isinstance(payload, Mapping) else {}
    settings = ctx.get("settings", {}) if isinstance(ctx, Mapping) else {}
    if isinstance(settings, Mapping) and settings.get("adapt_pool", None) is not None:
        value = str(settings.get("adapt_pool")).strip().lower()
        return value or None
    if isinstance(adapt_vqe, Mapping) and adapt_vqe.get("pool_type", None) is not None:
        value = str(adapt_vqe.get("pool_type")).strip().lower()
        return value or None
    return None


@dataclass(frozen=True)
class LockedImportedSubject:
    family: str
    pool_type: str | None
    subject_kind: str | None
    structure_locked: bool
    term_order_id: str | None
    operator_count: int | None
    runtime_term_count: int | None


def _detect_locked_imported_subject(ctx: Mapping[str, Any]) -> LockedImportedSubject | None:
    payload = ctx.get("payload", {}) if isinstance(ctx, Mapping) else {}
    adapt_vqe = payload.get("adapt_vqe", {}) if isinstance(payload, Mapping) else {}
    if isinstance(adapt_vqe, Mapping):
        fixed_meta = adapt_vqe.get("fixed_scaffold_metadata", {})
        route_family = None
        if isinstance(fixed_meta, Mapping):
            route_family = fixed_meta.get("route_family", None)
        if (
            bool(adapt_vqe.get("structure_locked", False))
            and str(adapt_vqe.get("fixed_scaffold_kind", "")).strip() != ""
            and str(route_family or "") == "locked_imported_scaffold_v1"
        ):
            return LockedImportedSubject(
                family="fixed_scaffold",
                pool_type=(
                    None
                    if adapt_vqe.get("pool_type", None) is None
                    else str(adapt_vqe.get("pool_type")).strip().lower()
                ),
                subject_kind=str(adapt_vqe.get("fixed_scaffold_kind")),
                structure_locked=True,
                term_order_id=(
                    None
                    if not isinstance(fixed_meta, Mapping) or fixed_meta.get("term_order_id", None) is None
                    else str(fixed_meta.get("term_order_id"))
                ),
                operator_count=(
                    None
                    if not isinstance(fixed_meta, Mapping) or fixed_meta.get("operator_count", None) is None
                    else int(fixed_meta.get("operator_count"))
                ),
                runtime_term_count=(
                    None
                    if not isinstance(fixed_meta, Mapping) or fixed_meta.get("runtime_term_count", None) is None
                    else int(fixed_meta.get("runtime_term_count"))
                ),
            )
    pool_type = _imported_pool_type(ctx)
    if str(pool_type) == "pareto_lean_l2":
        return LockedImportedSubject(
            family="legacy_pareto_lean_l2",
            pool_type=str(pool_type),
            subject_kind=None,
            structure_locked=True,
            term_order_id=None,
            operator_count=None,
            runtime_term_count=None,
        )
    return None


def _resolve_locked_imported_subject_context(
    artifact_json: str | Path,
    *,
    required_family: str,
    nonmatch_reason: str,
) -> tuple[dict[str, Any] | None, dict[str, Any], LockedImportedSubject | None, dict[str, Any] | None]:
    ctx = _load_imported_artifact_context(artifact_json)
    ansatz_input_state_meta = dict(ctx.get("ansatz_input_state_meta", {}))
    if not bool(ansatz_input_state_meta.get("available", False)):
        return (
            None,
            ansatz_input_state_meta,
            None,
            {
                "success": False,
                "available": False,
                "reason": str(
                    ansatz_input_state_meta.get("reason", "missing_ansatz_input_state_provenance")
                ),
                "source_kind": str(ctx.get("source_kind", "unknown")),
                "artifact_json": str(ctx["path"]),
                "structure_locked": False,
            },
        )
    subject = _detect_locked_imported_subject(ctx)
    if subject is None or str(subject.family) != str(required_family):
        return (
            None,
            ansatz_input_state_meta,
            subject,
            {
                "success": False,
                "available": False,
                "reason": str(nonmatch_reason),
                "source_kind": str(ctx.get("source_kind", "unknown")),
                "artifact_json": str(ctx["path"]),
                "pool_type": (None if subject is None else subject.pool_type),
                "subject_kind": (None if subject is None else subject.subject_kind),
                "structure_locked": bool(subject.structure_locked) if subject is not None else False,
            },
        )
    return ctx, ansatz_input_state_meta, subject, None


def _resolve_locked_imported_lean_context(
    artifact_json: str | Path,
    *,
    nonlean_reason: str,
) -> tuple[dict[str, Any] | None, dict[str, Any], str | None, dict[str, Any] | None]:
    ctx, ansatz_input_state_meta, subject, error_payload = _resolve_locked_imported_subject_context(
        artifact_json,
        required_family="legacy_pareto_lean_l2",
        nonmatch_reason=str(nonlean_reason),
    )
    return ctx, ansatz_input_state_meta, (None if subject is None else subject.pool_type), error_payload


def _resolve_locked_imported_fixed_scaffold_context(
    artifact_json: str | Path,
    *,
    nonfixed_reason: str,
) -> tuple[dict[str, Any] | None, dict[str, Any], LockedImportedSubject | None, dict[str, Any] | None]:
    return _resolve_locked_imported_subject_context(
        artifact_json,
        required_family="fixed_scaffold",
        nonmatch_reason=str(nonfixed_reason),
    )


def _bounded_append(history: list[dict[str, Any]], row: dict[str, Any], *, limit: int = 400) -> None:
    history.append(dict(row))
    if len(history) > int(limit):
        del history[:-int(limit)]


def _build_locked_imported_runtime_raw_oracle_config(
    *,
    shots: int,
    seed: int,
    oracle_repeats: int,
    oracle_aggregate: str,
    mitigation_config: Mapping[str, Any],
    symmetry_mitigation_config: Mapping[str, Any],
    backend_name: str | None,
    use_fake_backend: bool,
    allow_aer_fallback: bool,
    omp_shm_workaround: bool,
    seed_transpiler: int | None,
    transpile_optimization_level: int,
    runtime_profile_config: Mapping[str, Any] | None,
    runtime_session_config: Mapping[str, Any] | None,
    raw_transport: str = "auto",
    raw_store_memory: bool = False,
    raw_artifact_path: str | None = None,
) -> OracleConfig:
    noisy_mode = "backend_scheduled" if bool(use_fake_backend) else "runtime"
    return OracleConfig(
        noise_mode=str(noisy_mode),
        shots=int(shots),
        seed=int(seed),
        seed_transpiler=(None if seed_transpiler is None else int(seed_transpiler)),
        transpile_optimization_level=int(transpile_optimization_level),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        backend_name=(None if backend_name is None else str(backend_name)),
        use_fake_backend=bool(use_fake_backend),
        allow_aer_fallback=bool(allow_aer_fallback),
        aer_fallback_mode="sampler_shots",
        omp_shm_workaround=bool(omp_shm_workaround),
        mitigation=dict(normalize_mitigation_config(mitigation_config)),
        symmetry_mitigation=dict(normalize_symmetry_mitigation_config(symmetry_mitigation_config)),
        runtime_profile=dict(
            normalize_runtime_estimator_profile_config(runtime_profile_config)
        ),
        runtime_session=dict(
            normalize_runtime_session_policy_config(runtime_session_config)
        ),
        execution_surface="raw_measurement_v1",
        raw_transport=str(raw_transport),
        raw_store_memory=bool(raw_store_memory),
        raw_artifact_path=(None if raw_artifact_path in {None, ""} else str(raw_artifact_path)),
    )


def _build_locked_imported_ideal_oracle_config(
    *,
    shots: int,
    seed: int,
    oracle_repeats: int,
    oracle_aggregate: str,
) -> OracleConfig:
    return OracleConfig(
        noise_mode="ideal",
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        backend_name=None,
        use_fake_backend=False,
        mitigation={"mode": "none", "zne_scales": [], "dd_sequence": None, "local_readout_strategy": None},
        symmetry_mitigation={"mode": "off"},
    )


def _resolve_locked_imported_hh_symmetry_metadata(ctx: Mapping[str, Any]) -> dict[str, Any] | None:
    payload = ctx.get("payload", {}) if isinstance(ctx, Mapping) else {}
    settings = ctx.get("settings", {}) if isinstance(ctx, Mapping) else {}
    adapt_vqe = payload.get("adapt_vqe", {}) if isinstance(payload, Mapping) else {}
    fixed_meta = adapt_vqe.get("fixed_scaffold_metadata", {}) if isinstance(adapt_vqe, Mapping) else {}

    def _first_int(*values: Any) -> int | None:
        for raw in values:
            if raw in {None, ""}:
                continue
            if isinstance(raw, bool):
                continue
            try:
                return int(raw)
            except Exception:
                continue
        return None

    def _first_str(*values: Any) -> str | None:
        for raw in values:
            if raw in {None, ""}:
                continue
            token = str(raw).strip()
            if token != "":
                return token
        return None

    def _sector_from_pair(raw: Any) -> tuple[int | None, int | None]:
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) >= 2:
            try:
                return int(raw[0]), int(raw[1])
            except Exception:
                return None, None
        return None, None

    num_sites = _first_int(
        ctx.get("num_sites", None),
        settings.get("num_sites", None) if isinstance(settings, Mapping) else None,
        settings.get("L", None) if isinstance(settings, Mapping) else None,
        payload.get("num_sites", None) if isinstance(payload, Mapping) else None,
        payload.get("L", None) if isinstance(payload, Mapping) else None,
        adapt_vqe.get("num_sites", None) if isinstance(adapt_vqe, Mapping) else None,
        adapt_vqe.get("L", None) if isinstance(adapt_vqe, Mapping) else None,
        fixed_meta.get("num_sites", None) if isinstance(fixed_meta, Mapping) else None,
    )
    ordering = (
        _first_str(
            ctx.get("ordering", None),
            settings.get("ordering", None) if isinstance(settings, Mapping) else None,
            payload.get("ordering", None) if isinstance(payload, Mapping) else None,
            adapt_vqe.get("ordering", None) if isinstance(adapt_vqe, Mapping) else None,
            fixed_meta.get("ordering", None) if isinstance(fixed_meta, Mapping) else None,
        )
        or "blocked"
    )
    sector_from_ctx = _sector_from_pair(ctx.get("num_particles", None))
    sector_from_settings = _sector_from_pair(settings.get("num_particles", None) if isinstance(settings, Mapping) else None)
    sector_from_payload = _sector_from_pair(payload.get("num_particles", None) if isinstance(payload, Mapping) else None)
    sector_from_adapt = _sector_from_pair(adapt_vqe.get("num_particles", None) if isinstance(adapt_vqe, Mapping) else None)
    sector_n_up = _first_int(
        ctx.get("sector_n_up", None),
        settings.get("sector_n_up", None) if isinstance(settings, Mapping) else None,
        payload.get("sector_n_up", None) if isinstance(payload, Mapping) else None,
        adapt_vqe.get("sector_n_up", None) if isinstance(adapt_vqe, Mapping) else None,
        fixed_meta.get("sector_n_up", None) if isinstance(fixed_meta, Mapping) else None,
        sector_from_ctx[0],
        sector_from_settings[0],
        sector_from_payload[0],
        sector_from_adapt[0],
    )
    sector_n_dn = _first_int(
        ctx.get("sector_n_dn", None),
        settings.get("sector_n_dn", None) if isinstance(settings, Mapping) else None,
        payload.get("sector_n_dn", None) if isinstance(payload, Mapping) else None,
        adapt_vqe.get("sector_n_dn", None) if isinstance(adapt_vqe, Mapping) else None,
        fixed_meta.get("sector_n_dn", None) if isinstance(fixed_meta, Mapping) else None,
        sector_from_ctx[1],
        sector_from_settings[1],
        sector_from_payload[1],
        sector_from_adapt[1],
    )
    if num_sites is None:
        return None
    if sector_n_up is None or sector_n_dn is None:
        sector_n_up, sector_n_dn = _half_filled_particles(int(num_sites))
    return {
        "num_sites": int(num_sites),
        "ordering": str(ordering).strip().lower() or "blocked",
        "sector_n_up": int(sector_n_up),
        "sector_n_dn": int(sector_n_dn),
    }


def _build_locked_imported_energy_qop(
    *,
    ordered_labels_exyz: Sequence[str],
    static_coeff_map_exyz: Mapping[str, complex],
) -> SparsePauliOp:
    return build_time_dependent_sparse_qop(
        ordered_labels_exyz=list(ordered_labels_exyz),
        static_coeff_map_exyz=dict(static_coeff_map_exyz),
        drive_coeff_map_exyz=None,
    )


def _evaluate_locked_imported_circuit_ideal_energy(
    *,
    circuit: QuantumCircuit,
    ordered_labels_exyz: Sequence[str],
    static_coeff_map_exyz: Mapping[str, complex],
    shots: int,
    seed: int,
    oracle_repeats: int,
    oracle_aggregate: str,
    qop: SparsePauliOp | None = None,
) -> dict[str, Any]:
    qop_eval = (
        qop
        if qop is not None
        else _build_locked_imported_energy_qop(
            ordered_labels_exyz=ordered_labels_exyz,
            static_coeff_map_exyz=static_coeff_map_exyz,
        )
    )
    ideal_cfg = _build_locked_imported_ideal_oracle_config(
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
    )
    with ExpectationOracle(ideal_cfg) as ideal_oracle:
        ideal = ideal_oracle.evaluate(circuit, qop_eval)
    return {
        "ideal_mean": float(ideal.mean),
        "ideal_std": float(ideal.std),
        "ideal_stdev": float(ideal.stdev),
        "ideal_stderr": float(ideal.stderr),
    }


def _evaluate_locked_imported_circuit_noisy_energy(
    *,
    circuit: QuantumCircuit,
    ordered_labels_exyz: Sequence[str],
    static_coeff_map_exyz: Mapping[str, complex],
    shots: int,
    seed: int,
    oracle_repeats: int,
    oracle_aggregate: str,
    mitigation_config: Mapping[str, Any],
    symmetry_mitigation_config: Mapping[str, Any],
    backend_name: str | None,
    use_fake_backend: bool,
    allow_aer_fallback: bool,
    omp_shm_workaround: bool,
    seed_transpiler: int | None = None,
    transpile_optimization_level: int = 1,
    runtime_profile_config: Mapping[str, Any] | None = None,
    runtime_session_config: Mapping[str, Any] | None = None,
    runtime_job_observer: Any | None = None,
    runtime_trace_context: Mapping[str, Any] | None = None,
    qop: SparsePauliOp | None = None,
) -> dict[str, Any]:
    qop_eval = (
        qop
        if qop is not None
        else _build_locked_imported_energy_qop(
            ordered_labels_exyz=ordered_labels_exyz,
            static_coeff_map_exyz=static_coeff_map_exyz,
        )
    )
    noisy_mode = "backend_scheduled" if bool(use_fake_backend) else "runtime"
    noisy_cfg = OracleConfig(
        noise_mode=str(noisy_mode),
        shots=int(shots),
        seed=int(seed),
        seed_transpiler=(None if seed_transpiler is None else int(seed_transpiler)),
        transpile_optimization_level=int(transpile_optimization_level),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        backend_name=(None if backend_name is None else str(backend_name)),
        use_fake_backend=bool(use_fake_backend),
        allow_aer_fallback=bool(allow_aer_fallback),
        aer_fallback_mode="sampler_shots",
        omp_shm_workaround=bool(omp_shm_workaround),
        mitigation=dict(normalize_mitigation_config(mitigation_config)),
        symmetry_mitigation=dict(normalize_symmetry_mitigation_config(symmetry_mitigation_config)),
        runtime_profile=dict(
            normalize_runtime_estimator_profile_config(runtime_profile_config)
        ),
        runtime_session=dict(
            normalize_runtime_session_policy_config(runtime_session_config)
        ),
    )
    with ExpectationOracle(noisy_cfg) as noisy_oracle:
        noisy = noisy_oracle.evaluate(
            circuit,
            qop_eval,
            runtime_job_observer=runtime_job_observer,
            runtime_trace_context=runtime_trace_context,
        )
        backend_info = {
            "noise_mode": str(noisy_oracle.backend_info.noise_mode),
            "estimator_kind": str(noisy_oracle.backend_info.estimator_kind),
            "backend_name": noisy_oracle.backend_info.backend_name,
            "using_fake_backend": bool(noisy_oracle.backend_info.using_fake_backend),
            "details": dict(noisy_oracle.backend_info.details),
        }
    return {
        "noisy_mean": float(noisy.mean),
        "noisy_std": float(noisy.std),
        "noisy_stdev": float(noisy.stdev),
        "noisy_stderr": float(noisy.stderr),
        "backend_info": backend_info,
    }


def _combine_locked_imported_circuit_energy_evaluations(
    *,
    noisy_eval: Mapping[str, Any],
    ideal_eval: Mapping[str, Any],
    exact_energy: float | None = None,
) -> dict[str, Any]:
    def _coerce_optional_finite_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            out = float(value)
        except Exception:
            return None
        return out if math.isfinite(out) else None

    def _delta_payload(
        *,
        measured_mean: float,
        measured_stderr: float,
        target_mean: float,
        target_stderr: float,
    ) -> dict[str, float]:
        delta_mean = float(measured_mean - target_mean)
        delta_stderr = float(_combine_stderr(measured_stderr, target_stderr))
        return {
            "delta_mean": float(delta_mean),
            "delta_stderr": float(delta_stderr),
            "delta_abs": float(abs(delta_mean)),
        }

    noisy_mean = float(noisy_eval["noisy_mean"])
    noisy_stderr = float(noisy_eval["noisy_stderr"])
    ideal_mean = float(ideal_eval["ideal_mean"])
    ideal_stderr = float(ideal_eval["ideal_stderr"])
    ideal_delta = _delta_payload(
        measured_mean=float(noisy_mean),
        measured_stderr=float(noisy_stderr),
        target_mean=float(ideal_mean),
        target_stderr=float(ideal_stderr),
    )
    exact_mean = _coerce_optional_finite_float(exact_energy)
    exact_stderr = 0.0 if exact_mean is not None else None
    exact_delta = (
        _delta_payload(
            measured_mean=float(noisy_mean),
            measured_stderr=float(noisy_stderr),
            target_mean=float(exact_mean),
            target_stderr=0.0,
        )
        if exact_mean is not None
        else None
    )
    return {
        "noisy_mean": noisy_mean,
        "noisy_std": float(noisy_eval["noisy_std"]),
        "noisy_stdev": float(noisy_eval["noisy_stdev"]),
        "noisy_stderr": noisy_stderr,
        "ideal_mean": ideal_mean,
        "ideal_std": float(ideal_eval["ideal_std"]),
        "ideal_stdev": float(ideal_eval["ideal_stdev"]),
        "ideal_stderr": ideal_stderr,
        "exact_mean": exact_mean,
        "exact_stderr": exact_stderr,
        "energy_exact_mean": exact_mean,
        "energy_exact_stderr": exact_stderr,
        "delta_mean": float(ideal_delta["delta_mean"]),
        "delta_stderr": float(ideal_delta["delta_stderr"]),
        "delta_abs": float(ideal_delta["delta_abs"]),
        "delta_to_ideal_mean": float(ideal_delta["delta_mean"]),
        "delta_to_ideal_stderr": float(ideal_delta["delta_stderr"]),
        "delta_to_ideal_abs": float(ideal_delta["delta_abs"]),
        "delta_to_exact_mean": (
            None if exact_delta is None else float(exact_delta["delta_mean"])
        ),
        "delta_to_exact_stderr": (
            None if exact_delta is None else float(exact_delta["delta_stderr"])
        ),
        "delta_to_exact_abs": (
            None if exact_delta is None else float(exact_delta["delta_abs"])
        ),
        "backend_info": dict(noisy_eval.get("backend_info", {})),
    }


def _evaluate_locked_imported_circuit_energy(
    *,
    circuit: QuantumCircuit,
    ordered_labels_exyz: Sequence[str],
    static_coeff_map_exyz: Mapping[str, complex],
    shots: int,
    seed: int,
    oracle_repeats: int,
    oracle_aggregate: str,
    mitigation_config: Mapping[str, Any],
    symmetry_mitigation_config: Mapping[str, Any],
    backend_name: str | None,
    use_fake_backend: bool,
    allow_aer_fallback: bool,
    omp_shm_workaround: bool,
    seed_transpiler: int | None = None,
    transpile_optimization_level: int = 1,
    runtime_profile_config: Mapping[str, Any] | None = None,
    runtime_session_config: Mapping[str, Any] | None = None,
    runtime_job_observer: Any | None = None,
    runtime_trace_context: Mapping[str, Any] | None = None,
    exact_energy: float | None = None,
) -> dict[str, Any]:
    qop = _build_locked_imported_energy_qop(
        ordered_labels_exyz=ordered_labels_exyz,
        static_coeff_map_exyz=static_coeff_map_exyz,
    )
    ideal_eval = _evaluate_locked_imported_circuit_ideal_energy(
        circuit=circuit,
        ordered_labels_exyz=ordered_labels_exyz,
        static_coeff_map_exyz=static_coeff_map_exyz,
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        qop=qop,
    )
    noisy_eval = _evaluate_locked_imported_circuit_noisy_energy(
        circuit=circuit,
        ordered_labels_exyz=ordered_labels_exyz,
        static_coeff_map_exyz=static_coeff_map_exyz,
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        mitigation_config=mitigation_config,
        symmetry_mitigation_config=symmetry_mitigation_config,
        backend_name=backend_name,
        use_fake_backend=bool(use_fake_backend),
        allow_aer_fallback=bool(allow_aer_fallback),
        omp_shm_workaround=bool(omp_shm_workaround),
        seed_transpiler=seed_transpiler,
        transpile_optimization_level=int(transpile_optimization_level),
        runtime_profile_config=runtime_profile_config,
        runtime_session_config=runtime_session_config,
        runtime_job_observer=runtime_job_observer,
        runtime_trace_context=runtime_trace_context,
        qop=qop,
    )
    return _combine_locked_imported_circuit_energy_evaluations(
        noisy_eval=noisy_eval,
        ideal_eval=ideal_eval,
        exact_energy=exact_energy,
    )


def _build_locked_imported_fixed_theta_circuit_components(
    *,
    layout: Any,
    theta_runtime: np.ndarray,
    nq: int,
    ref_state: np.ndarray | None,
) -> dict[str, Any]:
    nq_i = int(nq)
    theta_arr = np.asarray(theta_runtime, dtype=float).reshape(-1)
    ref_arr = None if ref_state is None else np.asarray(ref_state, dtype=complex).reshape(-1)
    reference_circuit = QuantumCircuit(nq_i)
    if ref_arr is not None:
        _append_reference_state(reference_circuit, ref_arr)
    body_circuit = _build_ansatz_circuit(layout, theta_arr, nq_i, ref_state=None)
    full_circuit = reference_circuit.compose(body_circuit, inplace=False)
    return {
        "reference_circuit": reference_circuit,
        "body_circuit": body_circuit,
        "full_circuit": full_circuit,
    }


def _prepared_circuit_inverse_or_none(circuit: QuantumCircuit) -> QuantumCircuit | None:
    try:
        return circuit.inverse()
    except Exception:
        return None


def _build_local_zne_folded_circuit(
    *,
    reference_circuit: QuantumCircuit,
    body_circuit: QuantumCircuit,
    noise_scale: float,
) -> tuple[QuantumCircuit, dict[str, Any]]:
    scale_val = float(noise_scale)
    rounded = int(round(scale_val))
    if not math.isfinite(scale_val) or rounded < 1 or rounded % 2 == 0 or not math.isclose(
        scale_val, float(rounded), rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError(
            f"Local fixed-theta ZNE requires odd integer noise scales; got {noise_scale!r}."
        )
    if int(reference_circuit.num_qubits) != int(body_circuit.num_qubits):
        raise ValueError("Reference/body circuit qubit mismatch in local fixed-theta ZNE helper.")
    nq = int(reference_circuit.num_qubits)
    prepared_circuit = reference_circuit.compose(body_circuit, inplace=False)
    prepared_inverse = _prepared_circuit_inverse_or_none(prepared_circuit)
    folded = QuantumCircuit(nq)
    if prepared_inverse is not None:
        folded.compose(prepared_circuit, inplace=True)
        fold_unitary = prepared_circuit
        fold_inverse = prepared_inverse
        fold_scope = "prepared_circuit_full"
        fold_engine = "prepared_circuit_unitary_folding_v1"
        fold_warning = None
    else:
        folded.compose(reference_circuit, inplace=True)
        folded.compose(body_circuit, inplace=True)
        fold_unitary = body_circuit
        fold_inverse = body_circuit.inverse()
        fold_scope = "body_only"
        fold_engine = "body_unitary_folding_v1"
        fold_warning = "reference_state_prep_not_folded"
    fold_pairs = int((rounded - 1) // 2)
    for _ in range(fold_pairs):
        if nq > 0:
            folded.barrier(*range(nq))
        folded.compose(fold_inverse, inplace=True)
        if nq > 0:
            folded.barrier(*range(nq))
        folded.compose(fold_unitary, inplace=True)
    return folded, {
        "engine": str(fold_engine),
        "noise_scale": float(scale_val),
        "fold_pairs": int(fold_pairs),
        "zne_fold_scope": str(fold_scope),
        "warning": fold_warning,
    }


def _linear_zne_extrapolation(
    *,
    noise_scales: Sequence[float],
    noisy_means: Sequence[float],
    noisy_stderrs: Sequence[float],
) -> dict[str, Any]:
    x = np.asarray([float(v) for v in noise_scales], dtype=float)
    y = np.asarray([float(v) for v in noisy_means], dtype=float)
    s = np.asarray([float(v) for v in noisy_stderrs], dtype=float)
    if int(x.size) != int(y.size) or int(x.size) == 0:
        raise ValueError("Local fixed-theta ZNE extrapolation requires aligned non-empty scale/value arrays.")
    if int(x.size) == 1:
        return {
            "slope": 0.0,
            "intercept_mean": float(y[0]),
            "intercept_stderr": float(s[0]) if np.isfinite(s[0]) else float("nan"),
            "fit_kind": "degenerate_single_point",
        }
    weights = None
    if np.all(np.isfinite(s)) and np.all(s > 0.0):
        weights = 1.0 / s
    coeffs: np.ndarray
    cov: np.ndarray | None = None
    try:
        if weights is not None:
            coeffs, cov = np.polyfit(x, y, deg=1, w=weights, cov=True)
        else:
            coeffs, cov = np.polyfit(x, y, deg=1, cov=True)
    except Exception:
        coeffs = np.polyfit(x, y, deg=1)
        cov = None
    intercept_stderr = float("nan")
    if cov is not None and np.shape(cov) == (2, 2) and np.isfinite(cov[1, 1]) and cov[1, 1] >= 0.0:
        intercept_stderr = float(np.sqrt(float(cov[1, 1])))
    return {
        "slope": float(coeffs[0]),
        "intercept_mean": float(coeffs[1]),
        "intercept_stderr": float(intercept_stderr),
        "fit_kind": ("weighted_linear" if weights is not None else "linear"),
    }


def _evaluate_locked_imported_circuit_energy_local_zne(
    *,
    reference_circuit: QuantumCircuit,
    body_circuit: QuantumCircuit,
    ordered_labels_exyz: Sequence[str],
    static_coeff_map_exyz: Mapping[str, complex],
    shots: int,
    seed: int,
    oracle_repeats: int,
    oracle_aggregate: str,
    mitigation_config: Mapping[str, Any],
    symmetry_mitigation_config: Mapping[str, Any],
    backend_name: str | None,
    use_fake_backend: bool,
    allow_aer_fallback: bool,
    omp_shm_workaround: bool,
    seed_transpiler: int | None = None,
    transpile_optimization_level: int = 1,
    zne_scales: Sequence[float] = (1.0, 3.0, 5.0),
    ideal_eval: Mapping[str, Any] | None = None,
    exact_energy: float | None = None,
    qop: SparsePauliOp | None = None,
    compile_request_source: str = "fixed_scaffold_saved_theta_mitigation_matrix_cell",
) -> dict[str, Any]:
    qop_eval = (
        qop
        if qop is not None
        else _build_locked_imported_energy_qop(
            ordered_labels_exyz=ordered_labels_exyz,
            static_coeff_map_exyz=static_coeff_map_exyz,
        )
    )
    ideal_payload = (
        dict(ideal_eval)
        if isinstance(ideal_eval, Mapping)
        else _evaluate_locked_imported_circuit_ideal_energy(
            circuit=reference_circuit.compose(body_circuit, inplace=False),
            ordered_labels_exyz=ordered_labels_exyz,
            static_coeff_map_exyz=static_coeff_map_exyz,
            shots=int(shots),
            seed=int(seed),
            oracle_repeats=int(oracle_repeats),
            oracle_aggregate=str(oracle_aggregate),
            qop=qop_eval,
        )
    )
    compile_request = _build_compile_request_payload(
        backend_name=(None if backend_name in {None, ""} else str(backend_name)),
        seed_transpiler=(None if seed_transpiler is None else int(seed_transpiler)),
        transpile_optimization_level=int(transpile_optimization_level),
        source=str(compile_request_source),
    )
    per_factor_results: list[dict[str, Any]] = []
    base_factor_result: dict[str, Any] | None = None
    base_backend_info: dict[str, Any] = {}
    base_compile_observation: dict[str, Any] = {
        "available": False,
        "requested": dict(compile_request),
        "observed": None,
        "matches_requested": None,
        "mismatch_fields": [],
        "reason": "missing_base_factor",
    }
    base_compile_metrics: dict[str, Any] = {}

    for scale in [float(x) for x in zne_scales]:
        folded_circuit, folding_metadata = _build_local_zne_folded_circuit(
            reference_circuit=reference_circuit,
            body_circuit=body_circuit,
            noise_scale=float(scale),
        )
        noisy_eval = _evaluate_locked_imported_circuit_noisy_energy(
            circuit=folded_circuit,
            ordered_labels_exyz=ordered_labels_exyz,
            static_coeff_map_exyz=static_coeff_map_exyz,
            shots=int(shots),
            seed=int(seed),
            oracle_repeats=int(oracle_repeats),
            oracle_aggregate=str(oracle_aggregate),
            mitigation_config=mitigation_config,
            symmetry_mitigation_config=symmetry_mitigation_config,
            backend_name=backend_name,
            use_fake_backend=bool(use_fake_backend),
            allow_aer_fallback=bool(allow_aer_fallback),
            omp_shm_workaround=bool(omp_shm_workaround),
            seed_transpiler=seed_transpiler,
            transpile_optimization_level=int(transpile_optimization_level),
            qop=qop_eval,
        )
        combined = _combine_locked_imported_circuit_energy_evaluations(
            noisy_eval=noisy_eval,
            ideal_eval=ideal_payload,
            exact_energy=exact_energy,
        )
        backend_info = (
            dict(combined.get("backend_info", {}))
            if isinstance(combined.get("backend_info", {}), Mapping)
            else {}
        )
        compile_observation = _build_compile_observation_payload(
            requested=compile_request,
            backend_info=backend_info,
        )
        compile_metrics = _extract_compile_metrics_from_backend_info(backend_info)
        factor_result = {
            "noise_scale": float(scale),
            "folding_metadata": dict(folding_metadata),
            "compile_request": dict(compile_request),
            "compile_observation": dict(compile_observation),
            "transpile_seed": compile_metrics.get("transpile_seed", None),
            "transpile_optimization_level": compile_metrics.get(
                "transpile_optimization_level", None
            ),
            "compiled_two_qubit_count": int(compile_metrics.get("compiled_two_qubit_count", 0)),
            "compiled_depth": int(compile_metrics.get("compiled_depth", 0)),
            "compiled_size": int(compile_metrics.get("compiled_size", 0)),
            "backend_info": dict(backend_info),
            **dict(combined),
        }
        per_factor_results.append(factor_result)
        if base_factor_result is None or math.isclose(float(scale), 1.0, rel_tol=0.0, abs_tol=1e-9):
            base_factor_result = dict(factor_result)
            base_backend_info = dict(backend_info)
            base_compile_observation = dict(compile_observation)
            base_compile_metrics = dict(compile_metrics)

    if base_factor_result is None:
        raise RuntimeError("Local fixed-theta ZNE helper completed without a base factor result.")
    fit = _linear_zne_extrapolation(
        noise_scales=[float(rec["noise_scale"]) for rec in per_factor_results],
        noisy_means=[float(rec["noisy_mean"]) for rec in per_factor_results],
        noisy_stderrs=[float(rec["noisy_stderr"]) for rec in per_factor_results],
    )
    extrapolated_energy_mean = float(fit["intercept_mean"])
    extrapolated_energy_stderr = float(fit["intercept_stderr"])
    ideal_mean = float(ideal_payload["ideal_mean"])
    ideal_stderr = float(ideal_payload["ideal_stderr"])
    extrapolated_delta_mean = float(extrapolated_energy_mean - ideal_mean)
    extrapolated_delta_stderr = float(_combine_stderr(extrapolated_energy_stderr, ideal_stderr))
    try:
        exact_mean = (
            None
            if exact_energy is None
            else float(exact_energy)
            if math.isfinite(float(exact_energy))
            else None
        )
    except Exception:
        exact_mean = None
    exact_stderr = 0.0 if exact_mean is not None else None
    extrapolated_delta_to_exact_mean = (
        None if exact_mean is None else float(extrapolated_energy_mean - exact_mean)
    )
    extrapolated_delta_to_exact_stderr = (
        None
        if exact_mean is None
        else float(_combine_stderr(extrapolated_energy_stderr, 0.0))
    )
    base_folding_metadata = (
        dict(base_factor_result.get("folding_metadata", {}))
        if isinstance(base_factor_result.get("folding_metadata", {}), Mapping)
        else {}
    )
    return {
        "success": True,
        "zne_enabled": True,
        "zne_scales": [float(rec["noise_scale"]) for rec in per_factor_results],
        "per_factor_results": per_factor_results,
        "base_factor_result": dict(base_factor_result),
        "folding_metadata": dict(base_folding_metadata),
        "zne_fold_scope": base_folding_metadata.get("zne_fold_scope", None),
        "zne_fold_warning": base_folding_metadata.get("warning", None),
        "extrapolator": "linear",
        "extrapolator_fit": dict(fit),
        "extrapolated_energy_mean": float(extrapolated_energy_mean),
        "extrapolated_energy_stderr": float(extrapolated_energy_stderr),
        "noisy_mean": float(extrapolated_energy_mean),
        "noisy_stderr": float(extrapolated_energy_stderr),
        "ideal_mean": float(ideal_mean),
        "ideal_stderr": float(ideal_stderr),
        "exact_mean": exact_mean,
        "exact_stderr": exact_stderr,
        "energy_exact_mean": exact_mean,
        "energy_exact_stderr": exact_stderr,
        "delta_mean": float(extrapolated_delta_mean),
        "delta_stderr": float(extrapolated_delta_stderr),
        "delta_abs": float(abs(extrapolated_delta_mean)),
        "delta_to_ideal_mean": float(extrapolated_delta_mean),
        "delta_to_ideal_stderr": float(extrapolated_delta_stderr),
        "delta_to_ideal_abs": float(abs(extrapolated_delta_mean)),
        "delta_to_exact_mean": extrapolated_delta_to_exact_mean,
        "delta_to_exact_stderr": extrapolated_delta_to_exact_stderr,
        "delta_to_exact_abs": (
            None
            if extrapolated_delta_to_exact_mean is None
            else float(abs(extrapolated_delta_to_exact_mean))
        ),
        "compile_request": dict(compile_request),
        "compile_observation": dict(base_compile_observation),
        "matches_requested": base_compile_observation.get("matches_requested", None),
        "transpile_seed": base_compile_metrics.get("transpile_seed", None),
        "transpile_optimization_level": base_compile_metrics.get(
            "transpile_optimization_level", None
        ),
        "compiled_two_qubit_count": int(base_compile_metrics.get("compiled_two_qubit_count", 0)),
        "compiled_depth": int(base_compile_metrics.get("compiled_depth", 0)),
        "compiled_size": int(base_compile_metrics.get("compiled_size", 0)),
        "backend_info": dict(base_backend_info),
    }


def _evaluate_locked_imported_circuit_raw_energy(
    *,
    plan: Any,
    theta_runtime: np.ndarray,
    ordered_labels_exyz: Sequence[str],
    static_coeff_map_exyz: Mapping[str, complex],
    shots: int,
    seed: int,
    oracle_repeats: int,
    oracle_aggregate: str,
    mitigation_config: Mapping[str, Any],
    symmetry_mitigation_config: Mapping[str, Any],
    backend_name: str | None,
    use_fake_backend: bool,
    allow_aer_fallback: bool,
    omp_shm_workaround: bool,
    seed_transpiler: int | None = None,
    transpile_optimization_level: int = 1,
    runtime_profile_config: Mapping[str, Any] | None = None,
    runtime_session_config: Mapping[str, Any] | None = None,
    runtime_job_observer: Any | None = None,
    runtime_trace_context: Mapping[str, Any] | None = None,
    raw_transport: str = "auto",
    raw_store_memory: bool = False,
    raw_artifact_path: str | None = None,
) -> dict[str, Any]:
    qop = build_time_dependent_sparse_qop(
        ordered_labels_exyz=list(ordered_labels_exyz),
        static_coeff_map_exyz=dict(static_coeff_map_exyz),
        drive_coeff_map_exyz=None,
    )
    raw_cfg = _build_locked_imported_runtime_raw_oracle_config(
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        mitigation_config=mitigation_config,
        symmetry_mitigation_config=symmetry_mitigation_config,
        backend_name=backend_name,
        use_fake_backend=bool(use_fake_backend),
        allow_aer_fallback=bool(allow_aer_fallback),
        omp_shm_workaround=bool(omp_shm_workaround),
        seed_transpiler=seed_transpiler,
        transpile_optimization_level=int(transpile_optimization_level),
        runtime_profile_config=runtime_profile_config,
        runtime_session_config=runtime_session_config,
        raw_transport=str(raw_transport),
        raw_store_memory=bool(raw_store_memory),
        raw_artifact_path=(None if raw_artifact_path in {None, ""} else str(raw_artifact_path)),
    )
    ideal_cfg = _build_locked_imported_ideal_oracle_config(
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
    )
    with RawMeasurementOracle(raw_cfg) as raw_oracle, ExpectationOracle(ideal_cfg) as ideal_oracle:
        raw_bundle = raw_oracle.measure_observable(
            plan=plan,
            theta_runtime=np.asarray(theta_runtime, dtype=float),
            observable=qop,
            observable_family="fixed_scaffold_runtime_energy",
            semantic_tags={"route": "fixed_scaffold_runtime_raw_baseline"},
            runtime_job_observer=runtime_job_observer,
            runtime_trace_context=runtime_trace_context,
        )
        noisy = raw_bundle.estimate
        ideal_circuit = bind_parameterized_ansatz_circuit(plan, theta_runtime)
        ideal = ideal_oracle.evaluate(ideal_circuit, qop)
        backend_info = {
            "noise_mode": str(raw_cfg.noise_mode),
            "estimator_kind": "raw_measurement_oracle",
            "backend_name": raw_bundle.backend_snapshot.get("backend_name", backend_name),
            "using_fake_backend": bool(raw_cfg.use_fake_backend),
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
                "transpile_seed": (
                    None if seed_transpiler is None else int(seed_transpiler)
                ),
                "seed_transpiler": (
                    None if seed_transpiler is None else int(seed_transpiler)
                ),
                "transpile_optimization_level": int(transpile_optimization_level),
                "evaluation_id": str(raw_bundle.evaluation_id),
            },
        }
    return {
        "noisy_mean": float(noisy.mean),
        "noisy_std": float(noisy.std),
        "noisy_stdev": float(noisy.stdev),
        "noisy_stderr": float(noisy.stderr),
        "ideal_mean": float(ideal.mean),
        "ideal_std": float(ideal.std),
        "ideal_stdev": float(ideal.stdev),
        "ideal_stderr": float(ideal.stderr),
        "delta_mean": float(noisy.mean - ideal.mean),
        "delta_stderr": float(_combine_stderr(noisy.stderr, ideal.stderr)),
        "backend_info": backend_info,
    }


def _evaluate_locked_imported_circuit_raw_symmetry_diagnostic(
    *,
    plan: Any,
    theta_runtime: np.ndarray,
    shots: int,
    seed: int,
    oracle_repeats: int,
    oracle_aggregate: str,
    mitigation_config: Mapping[str, Any],
    symmetry_mitigation_config: Mapping[str, Any],
    backend_name: str | None,
    use_fake_backend: bool,
    allow_aer_fallback: bool,
    omp_shm_workaround: bool,
    num_sites: int | None,
    ordering: str | None,
    sector_n_up: int | None,
    sector_n_dn: int | None,
    seed_transpiler: int | None = None,
    transpile_optimization_level: int = 1,
    runtime_profile_config: Mapping[str, Any] | None = None,
    runtime_session_config: Mapping[str, Any] | None = None,
    runtime_job_observer: Any | None = None,
    runtime_trace_context: Mapping[str, Any] | None = None,
    raw_transport: str = "auto",
    raw_store_memory: bool = False,
    raw_artifact_path: str | None = None,
    diagonal_postprocessing_mitigation_config: Mapping[str, Any] | None = None,
    diagonal_postprocessing_symmetry_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if num_sites is None or sector_n_up is None or sector_n_dn is None:
        return {
            "success": False,
            "available": False,
            "reason": "missing_target_sector_metadata",
            "observable_family": "fixed_scaffold_runtime_all_z_symmetry_diagnostic",
            "evaluation_id": None,
            "transport": None,
            "raw_artifact_path": (None if raw_artifact_path in {None, ""} else str(raw_artifact_path)),
            "record_count": 0,
            "group_count": None,
            "term_count": None,
            "compile_signatures_by_basis": {},
            "summary": None,
            "diagonal_postprocessing": None,
            "error_type": None,
            "error_message": None,
        }
    raw_cfg = _build_locked_imported_runtime_raw_oracle_config(
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        mitigation_config=mitigation_config,
        symmetry_mitigation_config=symmetry_mitigation_config,
        backend_name=backend_name,
        use_fake_backend=bool(use_fake_backend),
        allow_aer_fallback=bool(allow_aer_fallback),
        omp_shm_workaround=bool(omp_shm_workaround),
        seed_transpiler=seed_transpiler,
        transpile_optimization_level=int(transpile_optimization_level),
        runtime_profile_config=runtime_profile_config,
        runtime_session_config=runtime_session_config,
        raw_transport=str(raw_transport),
        raw_store_memory=bool(raw_store_memory),
        raw_artifact_path=(None if raw_artifact_path in {None, ""} else str(raw_artifact_path)),
    )
    try:
        with RawMeasurementOracle(raw_cfg) as raw_oracle:
            bundle = raw_oracle.measure_observable(
                plan=plan,
                theta_runtime=np.asarray(theta_runtime, dtype=float),
                observable=_all_z_full_register_qop(int(getattr(plan, "nq"))),
                observable_family="fixed_scaffold_runtime_all_z_symmetry_diagnostic",
                semantic_tags={
                    "route": "fixed_scaffold_runtime_raw_baseline",
                    "diagnostic_kind": "all_z_full_register_v1",
                    "symmetry_num_sites": int(num_sites),
                    "symmetry_ordering": str(ordering or "blocked"),
                    "symmetry_sector_n_up": int(sector_n_up),
                    "symmetry_sector_n_dn": int(sector_n_dn),
                },
                runtime_job_observer=runtime_job_observer,
                runtime_trace_context=runtime_trace_context,
            )
            diagonal_postprocessing = (
                _summarize_hh_full_register_z_records_postprocessed(
                    getattr(bundle, "records", ()),
                    backend_target=raw_oracle.backend_target,
                    mitigation_config=(
                        diagonal_postprocessing_mitigation_config
                        if diagonal_postprocessing_mitigation_config is not None
                        else {"mode": "none"}
                    ),
                    symmetry_mitigation_config=(
                        diagonal_postprocessing_symmetry_config
                        if diagonal_postprocessing_symmetry_config is not None
                        else {"mode": "off"}
                    ),
                    num_sites=int(num_sites),
                    ordering=str(ordering or "blocked"),
                    sector_n_up=int(sector_n_up),
                    sector_n_dn=int(sector_n_dn),
                    expected_repeat_count=int(oracle_repeats),
                    shots=int(shots),
                )
                if bool(use_fake_backend)
                else {
                    "success": False,
                    "available": False,
                    "reason": "local_fake_backend_only",
                    "summary": None,
                    "readout_details": None,
                    "error_type": None,
                    "error_message": None,
                }
            )
        record_count = int(
            getattr(getattr(bundle, "estimate", None), "record_count", len(getattr(bundle, "records", ()) or ()))
        )
    except Exception as exc:
        return {
            "success": False,
            "available": False,
            "reason": "measurement_failed",
            "observable_family": "fixed_scaffold_runtime_all_z_symmetry_diagnostic",
            "evaluation_id": None,
            "transport": None,
            "raw_artifact_path": (None if raw_artifact_path in {None, ""} else str(raw_artifact_path)),
            "record_count": 0,
            "group_count": None,
            "term_count": None,
            "compile_signatures_by_basis": {},
            "summary": None,
            "diagonal_postprocessing": None,
            "error_type": str(type(exc).__name__),
            "error_message": str(exc),
        }
    try:
        summary = _summarize_hh_full_register_z_records(
            getattr(bundle, "records", ()),
            num_sites=int(num_sites),
            ordering=str(ordering or "blocked"),
            sector_n_up=int(sector_n_up),
            sector_n_dn=int(sector_n_dn),
            expected_repeat_count=int(oracle_repeats),
        )
    except Exception as exc:
        return {
            "success": False,
            "available": False,
            "reason": "summary_failed",
            "observable_family": str(bundle.observable_family),
            "evaluation_id": str(bundle.evaluation_id),
            "transport": str(bundle.transport),
            "raw_artifact_path": bundle.raw_artifact_path,
            "record_count": int(record_count),
            "group_count": int(bundle.estimate.group_count),
            "term_count": int(bundle.estimate.term_count),
            "compile_signatures_by_basis": dict(bundle.compile_signatures_by_basis),
            "summary": None,
            "diagonal_postprocessing": dict(diagonal_postprocessing),
            "error_type": str(type(exc).__name__),
            "error_message": str(exc),
        }
    return {
        "success": True,
        "available": True,
        "reason": None,
        "observable_family": str(bundle.observable_family),
        "evaluation_id": str(bundle.evaluation_id),
        "transport": str(bundle.transport),
        "raw_artifact_path": bundle.raw_artifact_path,
        "record_count": int(record_count),
        "group_count": int(bundle.estimate.group_count),
        "term_count": int(bundle.estimate.term_count),
        "compile_signatures_by_basis": dict(bundle.compile_signatures_by_basis),
        "summary": dict(summary),
        "diagonal_postprocessing": dict(diagonal_postprocessing),
        "error_type": None,
        "error_message": None,
    }


def _evaluate_locked_imported_circuit_raw_symmetry_validation(
    *,
    plan: Any,
    theta_runtime: np.ndarray,
    symmetry_diagnostic: Mapping[str, Any] | None,
    num_sites: int | None,
    ordering: str | None,
    sector_n_up: int | None,
    sector_n_dn: int | None,
) -> dict[str, Any]:
    if num_sites is None or sector_n_up is None or sector_n_dn is None:
        return {
            "success": False,
            "available": False,
            "reason": "missing_target_sector_metadata",
            "reference_source": None,
            "metrics": {},
            "notes": ["diagnostic_only", "no_energy_correction"],
            "error_type": None,
            "error_message": None,
        }
    if not isinstance(symmetry_diagnostic, Mapping) or not bool(symmetry_diagnostic.get("available", False)):
        return {
            "success": False,
            "available": False,
            "reason": "symmetry_diagnostic_unavailable",
            "reference_source": None,
            "metrics": {},
            "notes": ["diagnostic_only", "no_energy_correction"],
            "error_type": None,
            "error_message": None,
        }
    raw_summary = (
        symmetry_diagnostic.get("summary", {})
        if isinstance(symmetry_diagnostic.get("summary", {}), Mapping)
        else {}
    )

    def _maybe_float(raw: Any) -> float | None:
        if raw in {None, ""}:
            return None
        try:
            return float(raw)
        except Exception:
            return None

    try:
        circuit = bind_parameterized_ansatz_circuit(plan, np.asarray(theta_runtime, dtype=float))
        reference = _summarize_hh_exact_diagonal_reference(
            circuit,
            num_sites=int(num_sites),
            ordering=str(ordering or "blocked"),
            sector_n_up=int(sector_n_up),
            sector_n_dn=int(sector_n_dn),
        )
    except Exception as exc:
        return {
            "success": False,
            "available": False,
            "reason": "reference_unavailable",
            "reference_source": "ideal_diagonal_v1",
            "metrics": {},
            "notes": ["diagnostic_only", "no_energy_correction"],
            "error_type": str(type(exc).__name__),
            "error_message": str(exc),
        }

    def _metric(raw_mean_key: str, raw_stderr_key: str, ref_key: str) -> dict[str, Any]:
        raw_mean = _maybe_float(raw_summary.get(raw_mean_key, None))
        raw_stderr = _maybe_float(raw_summary.get(raw_stderr_key, None))
        reference_value = _maybe_float(reference.get(ref_key, None))
        delta = (
            None
            if raw_mean is None or reference_value is None
            else float(raw_mean - reference_value)
        )
        return {
            "raw_mean": raw_mean,
            "raw_stderr": raw_stderr,
            "reference": reference_value,
            "delta": delta,
        }

    return {
        "success": True,
        "available": True,
        "reason": None,
        "reference_source": str(reference.get("source", "ideal_diagonal_v1")),
        "target_sector": dict(reference.get("target_sector", {})),
        "metrics": {
            "sector_weight": _metric(
                "sector_weight_mean",
                "sector_weight_stderr",
                "sector_weight",
            ),
            "doublon_total": _metric(
                "doublon_total_mean",
                "doublon_total_stderr",
                "doublon_total",
            ),
        },
        "notes": ["diagnostic_only", "no_energy_correction"],
        "error_type": None,
        "error_message": None,
    }


def _evaluate_locked_imported_circuit_raw_diagonal_postprocessing_validation(
    *,
    plan: Any,
    theta_runtime: np.ndarray,
    diagonal_postprocessing: Mapping[str, Any] | None,
    num_sites: int | None,
    ordering: str | None,
    sector_n_up: int | None,
    sector_n_dn: int | None,
) -> dict[str, Any]:
    if num_sites is None or sector_n_up is None or sector_n_dn is None:
        return {
            "success": False,
            "available": False,
            "reason": "missing_target_sector_metadata",
            "reference_source": None,
            "metrics": {},
            "notes": ["diagnostic_only", "diagonal_only"],
            "error_type": None,
            "error_message": None,
        }
    if not isinstance(diagonal_postprocessing, Mapping) or not bool(
        diagonal_postprocessing.get("available", False)
    ):
        return {
            "success": False,
            "available": False,
            "reason": "diagonal_postprocessing_unavailable",
            "reference_source": None,
            "metrics": {},
            "notes": ["diagnostic_only", "diagonal_only"],
            "error_type": None,
            "error_message": None,
        }
    summary = (
        diagonal_postprocessing.get("summary", {})
        if isinstance(diagonal_postprocessing.get("summary", {}), Mapping)
        else {}
    )

    def _maybe_float(raw: Any) -> float | None:
        if raw in {None, ""}:
            return None
        try:
            return float(raw)
        except Exception:
            return None

    try:
        circuit = bind_parameterized_ansatz_circuit(plan, np.asarray(theta_runtime, dtype=float))
        reference = _summarize_hh_exact_diagonal_reference(
            circuit,
            num_sites=int(num_sites),
            ordering=str(ordering or "blocked"),
            sector_n_up=int(sector_n_up),
            sector_n_dn=int(sector_n_dn),
        )
    except Exception as exc:
        return {
            "success": False,
            "available": False,
            "reason": "reference_unavailable",
            "reference_source": "ideal_diagonal_v1",
            "metrics": {},
            "notes": ["diagnostic_only", "diagonal_only"],
            "error_type": str(type(exc).__name__),
            "error_message": str(exc),
        }

    def _metric(observed_mean_key: str, observed_stderr_key: str, ref_key: str) -> dict[str, Any]:
        observed_mean = _maybe_float(summary.get(observed_mean_key, None))
        observed_stderr = _maybe_float(summary.get(observed_stderr_key, None))
        reference_value = _maybe_float(reference.get(ref_key, None))
        delta = (
            None
            if observed_mean is None or reference_value is None
            else float(observed_mean - reference_value)
        )
        return {
            "observed_mean": observed_mean,
            "observed_stderr": observed_stderr,
            "reference": reference_value,
            "delta": delta,
        }

    return {
        "success": True,
        "available": True,
        "reason": None,
        "reference_source": str(reference.get("source", "ideal_diagonal_v1")),
        "target_sector": dict(reference.get("target_sector", {})),
        "metrics": {
            "sector_weight": _metric(
                "sector_weight_mean",
                "sector_weight_stderr",
                "sector_weight",
            ),
            "doublon_total": _metric(
                "doublon_total_mean",
                "doublon_total_stderr",
                "doublon_total",
            ),
        },
        "notes": ["diagnostic_only", "diagonal_only"],
        "error_type": None,
        "error_message": None,
    }


def _evaluate_locked_imported_circuit_raw_symmetry_bootstrap(
    *,
    symmetry_diagnostic: Mapping[str, Any] | None,
    num_sites: int | None,
    ordering: str | None,
    sector_n_up: int | None,
    sector_n_dn: int | None,
    oracle_repeats: int,
    seed: int,
) -> dict[str, Any]:
    if num_sites is None or sector_n_up is None or sector_n_dn is None:
        return {
            "success": False,
            "available": False,
            "reason": "missing_target_sector_metadata",
            "summary": None,
            "error_type": None,
            "error_message": None,
        }
    if not isinstance(symmetry_diagnostic, Mapping) or not bool(symmetry_diagnostic.get("available", False)):
        return {
            "success": False,
            "available": False,
            "reason": "symmetry_diagnostic_unavailable",
            "summary": None,
            "error_type": None,
            "error_message": None,
        }
    raw_artifact_path = symmetry_diagnostic.get("raw_artifact_path", None)
    evaluation_id = symmetry_diagnostic.get("evaluation_id", None)
    observable_family = symmetry_diagnostic.get("observable_family", None)
    if raw_artifact_path in {None, ""} or evaluation_id in {None, ""}:
        return {
            "success": False,
            "available": False,
            "reason": "raw_artifact_unavailable",
            "summary": None,
            "error_type": None,
            "error_message": None,
        }
    try:
        summary = _bootstrap_hh_full_register_z_artifact(
            str(raw_artifact_path),
            evaluation_id=str(evaluation_id),
            observable_family=(None if observable_family in {None, ""} else str(observable_family)),
            num_sites=int(num_sites),
            ordering=str(ordering or "blocked"),
            sector_n_up=int(sector_n_up),
            sector_n_dn=int(sector_n_dn),
            expected_repeat_count=int(oracle_repeats),
            seed=int(seed),
        )
    except Exception as exc:
        return {
            "success": False,
            "available": False,
            "reason": "bootstrap_failed",
            "summary": None,
            "error_type": str(type(exc).__name__),
            "error_message": str(exc),
        }
    return {
        "success": True,
        "available": True,
        "reason": None,
        "summary": dict(summary),
        "error_type": None,
        "error_message": None,
    }


def _extract_compile_metrics_from_backend_info(backend_info: Mapping[str, Any] | None) -> dict[str, Any]:
    details = (
        backend_info.get("details", {})
        if isinstance(backend_info, Mapping)
        and isinstance(backend_info.get("details", {}), Mapping)
        else {}
    )
    transpile_seed_raw = details.get("transpile_seed", details.get("seed_transpiler", None))
    return {
        "transpile_seed": (
            None if transpile_seed_raw is None else int(transpile_seed_raw)
        ),
        "transpile_optimization_level": int(details.get("transpile_optimization_level", 1)),
        "compiled_two_qubit_count": int(details.get("compiled_two_qubit_count", 0)),
        "compiled_cx_count": int(details.get("compiled_cx_count", 0)),
        "compiled_ecr_count": int(details.get("compiled_ecr_count", 0)),
        "compiled_depth": int(details.get("compiled_depth", 0)),
        "compiled_size": int(details.get("compiled_size", 0)),
        "layout_physical_qubits": [
            int(x) for x in list(details.get("layout_physical_qubits", []))
        ],
        "compiled_num_qubits": int(details.get("compiled_num_qubits", 0)),
        "compiled_op_counts": (
            dict(details.get("compiled_op_counts", {}))
            if isinstance(details.get("compiled_op_counts", {}), Mapping)
            else {}
        ),
    }


def _build_compile_request_payload(
    *,
    backend_name: str | None,
    seed_transpiler: int | None,
    transpile_optimization_level: int | None,
    source: str,
) -> dict[str, Any]:
    return {
        "backend_name": (None if backend_name in {None, ""} else str(backend_name)),
        "seed_transpiler": (
            None if seed_transpiler is None else int(seed_transpiler)
        ),
        "transpile_optimization_level": (
            None
            if transpile_optimization_level is None
            else int(transpile_optimization_level)
        ),
        "source": str(source),
    }


def _build_compile_observation_payload(
    *,
    requested: Mapping[str, Any],
    backend_info: Mapping[str, Any] | None,
) -> dict[str, Any]:
    details = (
        backend_info.get("details", {})
        if isinstance(backend_info, Mapping)
        and isinstance(backend_info.get("details", {}), Mapping)
        else {}
    )
    compile_available = bool(
        details
        and any(
            key in details
            for key in (
                "transpile_seed",
                "seed_transpiler",
                "transpile_optimization_level",
                "layout_physical_qubits",
                "compiled_num_qubits",
                "compiled_two_qubit_count",
                "compiled_depth",
                "compiled_size",
            )
        )
    )
    if not compile_available:
        return {
            "available": False,
            "requested": dict(requested),
            "observed": None,
            "matches_requested": None,
            "mismatch_fields": [],
            "reason": "compile_metrics_unavailable",
        }
    metrics = _extract_compile_metrics_from_backend_info(backend_info)
    observed = {
        "backend_name": (
            None
            if not isinstance(backend_info, Mapping)
            or backend_info.get("backend_name", None) in {None, ""}
            else str(backend_info.get("backend_name"))
        ),
        "seed_transpiler": metrics.get("transpile_seed", None),
        "transpile_optimization_level": metrics.get(
            "transpile_optimization_level", None
        ),
        "layout_physical_qubits": [
            int(x) for x in metrics.get("layout_physical_qubits", [])
        ],
        "compiled_num_qubits": metrics.get("compiled_num_qubits", None),
        "compiled_two_qubit_count": metrics.get("compiled_two_qubit_count", None),
        "compiled_depth": metrics.get("compiled_depth", None),
        "compiled_size": metrics.get("compiled_size", None),
    }
    mismatch_fields: list[str] = []
    if requested.get("backend_name", None) not in {None, ""} and observed["backend_name"] != str(
        requested.get("backend_name")
    ):
        mismatch_fields.append("backend_name")
    if requested.get("seed_transpiler", None) is not None and observed[
        "seed_transpiler"
    ] != int(requested.get("seed_transpiler")):
        mismatch_fields.append("seed_transpiler")
    if requested.get("transpile_optimization_level", None) is not None and observed[
        "transpile_optimization_level"
    ] != int(requested.get("transpile_optimization_level")):
        mismatch_fields.append("transpile_optimization_level")
    return {
        "available": True,
        "requested": dict(requested),
        "observed": dict(observed),
        "matches_requested": bool(len(mismatch_fields) == 0),
        "mismatch_fields": list(mismatch_fields),
        "reason": None,
    }


def _rank_imported_compile_control_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    rank_policy: str,
    delta_field: str = "delta_abs",
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = [dict(x) for x in candidates if isinstance(x, Mapping)]
    policy = str(rank_policy).strip().lower()
    if policy != "delta_mean_then_two_qubit_then_depth_then_size":
        raise ValueError(
            "Unsupported fixed lean compile-control scout rank policy "
            f"{rank_policy!r}."
        )

    primary_delta_field = str(delta_field).strip() or "delta_abs"

    def _delta_abs_value(rec: Mapping[str, Any]) -> float:
        primary = rec.get(primary_delta_field, None)
        if primary is not None:
            try:
                primary_val = float(primary)
            except Exception:
                primary_val = float("inf")
            if math.isfinite(primary_val):
                return float(primary_val)
        if primary_delta_field.endswith("_abs"):
            mean_field = f"{primary_delta_field[:-4]}_mean"
            mean_val = rec.get(mean_field, None)
            if mean_val is not None:
                try:
                    mean_float = float(mean_val)
                except Exception:
                    mean_float = float("inf")
                if math.isfinite(mean_float):
                    return float(abs(mean_float))
        fallback = rec.get("delta_abs", None)
        if fallback is not None:
            try:
                fallback_val = float(fallback)
            except Exception:
                fallback_val = float("inf")
            if math.isfinite(fallback_val):
                return float(fallback_val)
        delta_mean = rec.get("delta_mean", None)
        if delta_mean is None:
            return float("inf")
        try:
            delta_mean_val = float(delta_mean)
        except Exception:
            return float("inf")
        return float(abs(delta_mean_val)) if math.isfinite(delta_mean_val) else float("inf")

    def _sort_key(rec: Mapping[str, Any]) -> tuple[float, int, int, int, int, int]:
        return (
            _delta_abs_value(rec),
            int(rec.get("compiled_two_qubit_count", 10**9)),
            int(rec.get("compiled_depth", 10**9)),
            int(rec.get("compiled_size", 10**9)),
            int(rec.get("transpile_optimization_level", 10**9)),
            int(rec.get("transpile_seed", 10**9)),
        )

    ranked.sort(key=_sort_key)
    return ranked


def _runtime_compile_signature(backend_info: Mapping[str, Any] | None) -> dict[str, Any]:
    metrics = _extract_compile_metrics_from_backend_info(backend_info)
    return {
        "transpile_seed": metrics.get("transpile_seed", None),
        "transpile_optimization_level": metrics.get("transpile_optimization_level", None),
        "layout_physical_qubits": [int(x) for x in metrics.get("layout_physical_qubits", [])],
        "compiled_num_qubits": int(metrics.get("compiled_num_qubits", 0)),
        "compiled_two_qubit_count": int(metrics.get("compiled_two_qubit_count", 0)),
        "compiled_depth": int(metrics.get("compiled_depth", 0)),
        "compiled_size": int(metrics.get("compiled_size", 0)),
    }


def _runtime_compile_signature_matches(
    lhs: Mapping[str, Any] | None,
    rhs: Mapping[str, Any] | None,
) -> bool:
    return dict(_runtime_compile_signature(lhs)) == dict(_runtime_compile_signature(rhs))


def _run_locked_imported_runtime_phase_evals(
    *,
    circuit: QuantumCircuit,
    ordered_labels_exyz: Sequence[str],
    static_coeff_map_exyz: Mapping[str, complex],
    shots: int,
    seed: int,
    oracle_repeats: int,
    oracle_aggregate: str,
    mitigation_config: Mapping[str, Any],
    symmetry_mitigation_config: Mapping[str, Any],
    backend_name: str | None,
    allow_aer_fallback: bool,
    omp_shm_workaround: bool,
    seed_transpiler: int | None,
    transpile_optimization_level: int,
    runtime_session_config: Mapping[str, Any] | None,
    main_runtime_profile_config: Mapping[str, Any],
    dd_probe_runtime_profile_config: Mapping[str, Any] | None = None,
    final_audit_runtime_profile_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    main_eval = _evaluate_locked_imported_circuit_energy(
        circuit=circuit,
        ordered_labels_exyz=ordered_labels_exyz,
        static_coeff_map_exyz=static_coeff_map_exyz,
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        mitigation_config=mitigation_config,
        symmetry_mitigation_config=symmetry_mitigation_config,
        backend_name=backend_name,
        use_fake_backend=False,
        allow_aer_fallback=bool(allow_aer_fallback),
        omp_shm_workaround=bool(omp_shm_workaround),
        seed_transpiler=seed_transpiler,
        transpile_optimization_level=int(transpile_optimization_level),
        runtime_profile_config=main_runtime_profile_config,
        runtime_session_config=runtime_session_config,
    )
    baseline_backend = dict(main_eval.get("backend_info", {}))
    baseline_signature = _runtime_compile_signature(baseline_backend)
    payload: dict[str, Any] = {
        "main": {
            "success": True,
            "profile": dict(normalize_runtime_estimator_profile_config(main_runtime_profile_config)),
            "evaluation": dict(main_eval),
            "compile_signature": dict(baseline_signature),
        }
    }
    for phase_name, profile_cfg in (
        ("dd_probe", dd_probe_runtime_profile_config),
        ("final_audit_zne", final_audit_runtime_profile_config),
    ):
        if profile_cfg is None:
            payload[phase_name] = {"enabled": False, "success": False, "reason": f"{phase_name}_disabled"}
            continue
        phase_eval = _evaluate_locked_imported_circuit_energy(
            circuit=circuit,
            ordered_labels_exyz=ordered_labels_exyz,
            static_coeff_map_exyz=static_coeff_map_exyz,
            shots=int(shots),
            seed=int(seed),
            oracle_repeats=int(oracle_repeats),
            oracle_aggregate=str(oracle_aggregate),
            mitigation_config=mitigation_config,
            symmetry_mitigation_config=symmetry_mitigation_config,
            backend_name=backend_name,
            use_fake_backend=False,
            allow_aer_fallback=bool(allow_aer_fallback),
            omp_shm_workaround=bool(omp_shm_workaround),
            seed_transpiler=seed_transpiler,
            transpile_optimization_level=int(transpile_optimization_level),
            runtime_profile_config=profile_cfg,
            runtime_session_config=runtime_session_config,
        )
        phase_backend = dict(phase_eval.get("backend_info", {}))
        phase_signature = _runtime_compile_signature(phase_backend)
        if not _runtime_compile_signature_matches(baseline_backend, phase_backend):
            payload[phase_name] = {
                "enabled": True,
                "success": False,
                "reason": "compile_layout_drift",
                "profile": dict(normalize_runtime_estimator_profile_config(profile_cfg)),
                "expected_compile_signature": dict(baseline_signature),
                "observed_compile_signature": dict(phase_signature),
                "evaluation": dict(phase_eval),
            }
            continue
        payload[phase_name] = {
            "enabled": True,
            "success": True,
            "profile": dict(normalize_runtime_estimator_profile_config(profile_cfg)),
            "evaluation": dict(phase_eval),
            "compile_signature": dict(phase_signature),
        }
    return payload


def _build_reference_state(args: argparse.Namespace, num_particles: tuple[int, int]) -> np.ndarray:
    return _normalize_state(
        np.asarray(
            hubbard_holstein_reference_state(
                dims=int(args.L),
                num_particles=num_particles,
                n_ph_max=int(args.n_ph_max),
                boson_encoding=str(args.boson_encoding),
                indexing=str(args.ordering),
            ),
            dtype=complex,
        )
    )


def _build_hh_hva_ansatz(args: argparse.Namespace, *, reps: int) -> Any:
    return HubbardHolsteinPhysicalTermwiseAnsatz(
        dims=int(args.L),
        J=float(args.t),
        U=float(args.u),
        omega0=float(args.omega0),
        g=float(args.g_ep),
        n_ph_max=int(args.n_ph_max),
        boson_encoding=str(args.boson_encoding),
        reps=int(reps),
        repr_mode="JW",
        indexing=str(args.ordering),
        pbc=(str(args.boundary).strip().lower() == "periodic"),
    )


def _build_uccsd_fermion_lifted_pool(
    *,
    num_sites: int,
    num_particles: tuple[int, int],
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
) -> list[AnsatzTerm]:
    base = HardcodedUCCSDAnsatz(
        dims=int(num_sites),
        num_particles=(int(num_particles[0]), int(num_particles[1])),
        reps=1,
        repr_mode="JW",
        indexing=str(ordering),
        include_singles=True,
        include_doubles=True,
    )
    ferm_nq = 2 * int(num_sites)
    bps = int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
    boson_bits = int(num_sites) * bps
    nq_total = ferm_nq + boson_bits

    lifted_pool: list[AnsatzTerm] = []
    for op in base.base_terms:
        lifted = PauliPolynomial("JW")
        for term in op.polynomial.return_polynomial():
            coeff = complex(term.p_coeff)
            if abs(coeff) <= 1e-15:
                continue
            if abs(coeff.imag) > 1e-10:
                raise ValueError(f"UCCSD coeff has non-negligible imaginary part: {coeff}")
            ferm_ps = str(term.pw2strng())
            if len(ferm_ps) != int(ferm_nq):
                raise ValueError(
                    f"Unexpected UCCSD Pauli length {len(ferm_ps)} != ferm_nq {ferm_nq}."
                )
            full_ps = ("e" * int(boson_bits)) + ferm_ps
            lifted.add_term(PauliTerm(int(nq_total), ps=full_ps, pc=float(coeff.real)))
        lifted_pool.append(AnsatzTerm(label=f"uccsd_ferm_lifted::{op.label}", polynomial=lifted))

    return lifted_pool


def _build_hva_pool(args: argparse.Namespace) -> list[AnsatzTerm]:
    return list(_build_hh_hva_ansatz(args, reps=1).base_terms)


def _build_paop_full_pool(
    *,
    args: argparse.Namespace,
    num_particles: tuple[int, int],
) -> list[AnsatzTerm]:
    specs = make_paop_pool(
        "paop_full",
        num_sites=int(args.L),
        num_particles=(int(num_particles[0]), int(num_particles[1])),
        n_ph_max=int(args.n_ph_max),
        boson_encoding=str(args.boson_encoding),
        ordering=str(args.ordering),
        boundary=str(args.boundary),
        paop_r=int(args.paop_r),
        paop_split_paulis=bool(args.paop_split_paulis),
        paop_prune_eps=float(args.paop_prune_eps),
        paop_normalization=str(args.paop_normalization),
    )
    return [AnsatzTerm(label=str(lbl), polynomial=poly) for lbl, poly in specs]


def _run_vqe_stage_with_transition(
    *,
    stage_name: str,
    h_poly: Any,
    ansatz: Any,
    psi_ref: np.ndarray,
    exact_energy: float,
    restarts: int,
    seed: int,
    maxiter: int,
    method: str,
    transition_cfg: TransitionConfig,
) -> tuple[dict[str, Any], np.ndarray]:
    t0 = time.perf_counter()
    transition_state = TransitionState(cfg=transition_cfg)
    progress_events: list[dict[str, Any]] = []

    def _early_stop_checker(payload: dict[str, Any]) -> bool:
        energy_cur = payload.get("energy_current", None)
        if energy_cur is None:
            return False
        rec = update_transition_state(
            transition_state,
            delta_abs=abs(float(energy_cur) - float(exact_energy)),
        )
        progress_events.append(
            {
                "event": "objective_step",
                "restart_index": int(payload.get("restart_index", 0)),
                "nfev_so_far": int(payload.get("nfev_so_far", 0)),
                "energy_current": float(energy_cur),
                "delta_abs": float(rec["delta_abs"]),
                "slope": rec["slope"],
                "plateau_hits": int(rec["plateau_hits"]),
                "switch_triggered": bool(rec["switch_triggered"]),
            }
        )
        return bool(rec["switch_triggered"])

    def _progress_logger(ev: dict[str, Any]) -> None:
        evt = dict(ev)
        if "energy_current" in evt and evt["energy_current"] is not None:
            evt["delta_abs"] = float(abs(float(evt["energy_current"]) - float(exact_energy)))
        progress_events.append(evt)

    res = vqe_minimize(
        h_poly,
        ansatz,
        np.asarray(psi_ref, dtype=complex),
        restarts=int(restarts),
        seed=int(seed),
        maxiter=int(maxiter),
        method=str(method),
        progress_logger=_progress_logger,
        progress_every_s=0.0,
        progress_label=str(stage_name),
        track_history=False,
        emit_theta_in_progress=False,
        return_best_on_keyboard_interrupt=True,
        early_stop_checker=_early_stop_checker,
        energy_backend="one_apply_compiled",
    )

    theta = np.asarray(res.theta, dtype=float)
    psi_best = _normalize_state(np.asarray(ansatz.prepare_state(theta, psi_ref), dtype=complex).reshape(-1))
    final_energy = float(res.energy)

    if bool(transition_state.switch_triggered):
        stop_reason = "slope_plateau"
    elif bool(res.success):
        stop_reason = "optimizer_success"
    else:
        stop_reason = str(res.message)

    payload = {
        "stage": str(stage_name),
        "success": bool(res.success),
        "stop_reason": str(stop_reason),
        "optimizer_message": str(res.message),
        "energy": float(final_energy),
        "exact_energy": float(exact_energy),
        "delta_abs": float(abs(final_energy - float(exact_energy))),
        "num_parameters": int(getattr(ansatz, "num_parameters", theta.size)),
        "best_restart": int(res.best_restart + 1),
        "nfev": int(res.nfev),
        "nit": int(res.nit),
        "restarts": int(restarts),
        "maxiter": int(maxiter),
        "method": str(method),
        "theta": [float(x) for x in theta.tolist()],
        "transition": summarize_transition(transition_state),
        "progress_events": progress_events,
        "elapsed_s": float(time.perf_counter() - t0),
    }
    return payload, psi_best


def _run_adapt_stage_with_transition(
    *,
    h_poly: Any,
    psi_start: np.ndarray,
    pool: list[AnsatzTerm],
    exact_energy: float,
    allow_repeats: bool,
    max_depth: int,
    maxiter: int,
    eps_grad: float,
    eps_energy: float,
    seed: int,
    transition_cfg: TransitionConfig,
) -> tuple[dict[str, Any], np.ndarray]:
    t0 = time.perf_counter()
    if not pool:
        raise ValueError("Pool B is empty.")

    try:
        from scipy.optimize import minimize as scipy_minimize
    except Exception as exc:
        raise RuntimeError("SciPy is required for ADAPT stage in this report.") from exc

    rng = np.random.default_rng(int(seed))
    selected_ops: list[AnsatzTerm] = []
    theta = np.zeros(0, dtype=float)
    selected_executor: Any | None = None
    available_indices = set(range(len(pool)))
    history: list[dict[str, Any]] = []
    pauli_action_cache: dict[str, Any] = {}
    h_compiled = _compile_h_poly_shared(
        h_poly,
        tol=1e-12,
        pauli_action_cache=pauli_action_cache,
    )
    pool_compiled = [
        _compile_h_poly_shared(
            op.polynomial,
            tol=1e-12,
            pauli_action_cache=pauli_action_cache,
        )
        for op in pool
    ]
    nfev_total = 1
    nit_total = 0

    energy_current = float(
        _energy_one_apply_shared(np.asarray(psi_start, dtype=complex), h_poly, compiled=h_compiled)
    )
    transition_state = TransitionState(cfg=transition_cfg)
    update_transition_state(transition_state, delta_abs=abs(float(energy_current) - float(exact_energy)))

    stop_reason = "max_depth"

    for depth in range(int(max_depth)):
        candidate_indices = list(range(len(pool))) if bool(allow_repeats) else sorted(available_indices)
        if not candidate_indices:
            stop_reason = "pool_exhausted"
            break

        psi_current = _prepare_state_for_ansatz(
            np.asarray(psi_start, dtype=complex),
            selected_ops,
            theta,
            compiled_cache=selected_executor,
            normalize=True,
        )
        hpsi_current = _apply_pauli_polynomial(psi_current, h_poly, compiled=h_compiled)
        gradients = {
            idx: float(
                _commutator_gradient(
                    pool[idx],
                    psi_current,
                    h_poly,
                    compiled={
                        "h_compiled": h_compiled,
                        "pool_compiled": pool_compiled[idx],
                    },
                    h_psi_precomputed=hpsi_current,
                )
            )
            for idx in candidate_indices
        }
        grad_abs = {idx: abs(v) for idx, v in gradients.items()}
        best_idx = max(candidate_indices, key=lambda i: (grad_abs[i], -i))
        best_grad_abs = float(grad_abs[best_idx])

        if best_grad_abs < float(eps_grad):
            stop_reason = "eps_grad"
            break

        selected_ops.append(pool[best_idx])
        theta = np.append(theta, 0.0)
        selected_executor = _compile_ansatz_executor_shared(
            selected_ops,
            coefficient_tolerance=1e-12,
            ignore_identity=True,
            sort_terms=True,
            pauli_action_cache=pauli_action_cache,
        )
        if not bool(allow_repeats):
            available_indices.discard(best_idx)

        energy_prev = float(energy_current)
        def _obj(x: np.ndarray) -> float:
            psi_obj = _prepare_state_for_ansatz(
                np.asarray(psi_start, dtype=complex),
                selected_ops,
                np.asarray(x, dtype=float),
                compiled_cache=selected_executor,
                normalize=True,
            )
            return float(_energy_one_apply_shared(psi_obj, h_poly, compiled=h_compiled))

        x0 = np.asarray(theta, dtype=float) + 0.02 * rng.normal(size=theta.size)
        res = scipy_minimize(
            _obj,
            x0,
            method="COBYLA",
            options={"maxiter": int(maxiter), "rhobeg": 0.3},
        )
        theta = np.asarray(res.x, dtype=float)
        energy_current = float(res.fun)
        nfev_total += int(getattr(res, "nfev", 0))
        nit_total += int(getattr(res, "nit", 0))

        trans_rec = update_transition_state(
            transition_state,
            delta_abs=abs(float(energy_current) - float(exact_energy)),
        )

        history.append(
            {
                "depth": int(depth + 1),
                "selected_pool_index": int(best_idx),
                "selected_label": str(pool[best_idx].label),
                "max_gradient_abs": float(best_grad_abs),
                "energy_before_opt": float(energy_prev),
                "energy_after_opt": float(energy_current),
                "delta_energy_abs": float(abs(energy_current - energy_prev)),
                "delta_abs_vs_exact": float(abs(energy_current - float(exact_energy))),
                "slope": trans_rec["slope"],
                "plateau_hits": int(trans_rec["plateau_hits"]),
                "switch_triggered": bool(trans_rec["switch_triggered"]),
                "opt_nfev": int(getattr(res, "nfev", 0)),
                "opt_nit": int(getattr(res, "nit", 0)),
                "opt_success": bool(getattr(res, "success", False)),
                "opt_message": str(getattr(res, "message", "")),
            }
        )

        if abs(float(energy_current) - float(energy_prev)) < float(eps_energy):
            stop_reason = "eps_energy"
            break

        if bool(trans_rec["switch_triggered"]):
            stop_reason = "slope_plateau"
            break

    psi_best = _prepare_state_for_ansatz(
        np.asarray(psi_start, dtype=complex),
        selected_ops,
        theta,
        compiled_cache=selected_executor,
        normalize=True,
    )

    payload = {
        "stage": "adapt_pool_b",
        "success": True,
        "stop_reason": str(stop_reason),
        "energy": float(energy_current),
        "exact_energy": float(exact_energy),
        "delta_abs": float(abs(float(energy_current) - float(exact_energy))),
        "ansatz_depth": int(len(selected_ops)),
        "num_parameters": int(theta.size),
        "allow_repeats": bool(allow_repeats),
        "max_depth": int(max_depth),
        "maxiter": int(maxiter),
        "eps_grad": float(eps_grad),
        "eps_energy": float(eps_energy),
        "nfev_total": int(nfev_total),
        "nit_total": int(nit_total),
        "operators": [str(op.label) for op in selected_ops],
        "optimal_point": [float(x) for x in theta.tolist()],
        "history": history,
        "transition": summarize_transition(transition_state),
        "elapsed_s": float(time.perf_counter() - t0),
    }
    return payload, psi_best


def _parse_custom_s(raw: str | None, *, L: int) -> list[float] | None:
    if raw is None:
        return None
    txt = str(raw).strip()
    if not txt:
        return None
    if txt.startswith("["):
        data = json.loads(txt)
        arr = [float(x) for x in data]
    else:
        arr = [float(x.strip()) for x in txt.split(",") if x.strip()]
    if len(arr) != int(L):
        raise ValueError(f"drive-custom-s length {len(arr)} does not match L={L}")
    return arr


def _build_drive_profile(args: argparse.Namespace, *, enabled: bool) -> dict[str, Any] | None:
    if not bool(enabled):
        return None
    custom_weights = _parse_custom_s(args.drive_custom_s, L=int(args.L)) if str(args.drive_pattern) == "custom" else None
    return {
        "enabled": True,
        "A": float(args.drive_A),
        "omega": float(args.drive_omega),
        "tbar": float(args.drive_tbar),
        "phi": float(args.drive_phi),
        "pattern": str(args.drive_pattern),
        "custom_s": custom_weights,
        "include_identity": bool(args.drive_include_identity),
        "t0": float(args.drive_t0),
        "time_sampling": str(args.drive_time_sampling),
        "reference_steps_multiplier": int(args.exact_steps_multiplier),
        "reference_method": reference_method_name(str(args.drive_time_sampling)),
    }


def _drive_provider_from_profile(
    *,
    profile: dict[str, Any] | None,
    num_sites: int,
    nq_total: int,
    ordering: str,
) -> tuple[Any | None, dict[str, Any] | None]:
    if profile is None:
        return None, None
    drive = build_gaussian_sinusoid_density_drive(
        n_sites=int(num_sites),
        nq_total=int(nq_total),
        indexing=str(ordering),
        A=float(profile["A"]),
        omega=float(profile["omega"]),
        tbar=float(profile["tbar"]),
        phi=float(profile["phi"]),
        pattern_mode=str(profile["pattern"]),
        custom_weights=profile.get("custom_s"),
        include_identity=bool(profile.get("include_identity", False)),
        electron_qubit_offset=0,
        coeff_tol=0.0,
    )
    return drive.coeff_map_exyz, {
        "labels": int(len(drive.template.labels_exyz(include_identity=bool(drive.include_identity)))),
        "identity_included": bool(drive.include_identity),
    }


def _spin_orbital_bit_index(site: int, spin: int, num_sites: int, ordering: str) -> int:
    ord_norm = str(ordering).strip().lower()
    if ord_norm == "blocked":
        return int(site) if int(spin) == 0 else int(num_sites) + int(site)
    if ord_norm == "interleaved":
        return (2 * int(site)) + int(spin)
    raise ValueError(f"Unsupported ordering {ordering!r}")


def _staggered_qop(num_qubits: int, num_sites: int, ordering: str) -> SparsePauliOp:
    op = SparsePauliOp.from_list([("I" * int(num_qubits), 0.0)])
    L = int(num_sites)
    for site in range(L):
        sign = 1.0 if (site % 2 == 0) else -1.0
        up_idx = _spin_orbital_bit_index(site, 0, L, ordering)
        dn_idx = _spin_orbital_bit_index(site, 1, L, ordering)
        op = op + (sign / float(L)) * _number_operator_qop(int(num_qubits), int(up_idx))
        op = op + (sign / float(L)) * _number_operator_qop(int(num_qubits), int(dn_idx))
    return op.simplify(atol=1e-12)


def _time_sample(step_idx: int, dt: float, sampling: str) -> float:
    mode = str(sampling).strip().lower()
    if mode == "midpoint":
        return float((float(step_idx) + 0.5) * float(dt))
    if mode == "left":
        return float(float(step_idx) * float(dt))
    if mode == "right":
        return float((float(step_idx) + 1.0) * float(dt))
    raise ValueError(f"Unsupported time sampling {sampling!r}")


def _build_suzuki2_time_dependent_circuit(
    *,
    initial_circuit: QuantumCircuit,
    ordered_labels_exyz: list[str],
    static_coeff_map_exyz: dict[str, complex],
    drive_provider_exyz: Any | None,
    time_value: float,
    trotter_steps: int,
    drive_t0: float,
    drive_time_sampling: str,
) -> QuantumCircuit:
    qc = initial_circuit.copy()
    if abs(float(time_value)) <= 1e-15:
        return qc

    dt = float(time_value) / float(trotter_steps)
    synthesis = SuzukiTrotter(order=2, reps=1, preserve_order=True)

    for step_idx in range(int(trotter_steps)):
        t_sample = _time_sample(step_idx, dt, drive_time_sampling)
        drive_map = {}
        if drive_provider_exyz is not None:
            drive_map = dict(drive_provider_exyz(float(drive_t0) + float(t_sample)))
        qop = build_time_dependent_sparse_qop(
            ordered_labels_exyz=ordered_labels_exyz,
            static_coeff_map_exyz=static_coeff_map_exyz,
            drive_coeff_map_exyz=drive_map,
        )
        qc.append(
            PauliEvolutionGate(qop, time=float(dt), synthesis=synthesis),
            list(range(int(initial_circuit.num_qubits))),
        )
    return qc


def _build_cfqm_stage_map_exyz(
    *,
    ordered_labels_exyz: list[str],
    static_coeff_map_exyz: dict[str, complex],
    drive_maps_exyz: list[dict[str, complex]],
    a_row: list[float],
    s_static: float,
    coeff_drop_abs_tol: float,
) -> dict[str, complex]:
    ordered_set = set(ordered_labels_exyz)
    stage_map: dict[str, complex] = {}

    for lbl in ordered_labels_exyz:
        coeff0 = static_coeff_map_exyz.get(lbl, 0.0 + 0.0j)
        scaled = complex(float(s_static)) * complex(coeff0)
        if scaled != 0.0:
            stage_map[lbl] = scaled

    for j, drive_map in enumerate(drive_maps_exyz):
        w = float(a_row[j])
        if w == 0.0:
            continue
        for lbl, coeff_drive in drive_map.items():
            if lbl not in ordered_set:
                continue
            inc = complex(w) * complex(coeff_drive)
            if inc == 0.0 and lbl not in stage_map:
                continue
            stage_map[lbl] = stage_map.get(lbl, 0.0 + 0.0j) + inc

    drop = float(max(0.0, coeff_drop_abs_tol))
    if drop > 0.0:
        for lbl in list(stage_map):
            if abs(stage_map[lbl]) < drop:
                del stage_map[lbl]
    return stage_map


def _build_cfqm_time_dependent_circuit(
    *,
    method: str,
    initial_circuit: QuantumCircuit,
    ordered_labels_exyz: list[str],
    static_coeff_map_exyz: dict[str, complex],
    drive_provider_exyz: Any | None,
    time_value: float,
    trotter_steps: int,
    drive_t0: float,
    coeff_drop_abs_tol: float,
) -> QuantumCircuit:
    qc = initial_circuit.copy()
    if abs(float(time_value)) <= 1e-15:
        return qc

    scheme = get_cfqm_scheme(str(method))
    c_nodes = [float(x) for x in scheme["c"]]
    a_rows = [[float(v) for v in row] for row in scheme["a"]]
    s_static = [float(v) for v in scheme["s_static"]]

    dt = float(time_value) / float(trotter_steps)
    synthesis = SuzukiTrotter(order=2, reps=1, preserve_order=True)
    qubits = list(range(int(initial_circuit.num_qubits)))

    for step_idx in range(int(trotter_steps)):
        t_abs = float(drive_t0) + float(step_idx) * float(dt)
        drive_maps_exyz: list[dict[str, complex]] = []
        for c_j in c_nodes:
            t_node = float(t_abs) + float(c_j) * float(dt)
            raw = {} if drive_provider_exyz is None else dict(drive_provider_exyz(float(t_node)))
            drive_maps_exyz.append({str(k): complex(v) for k, v in raw.items()})

        for k, a_row in enumerate(a_rows):
            stage_map = _build_cfqm_stage_map_exyz(
                ordered_labels_exyz=list(ordered_labels_exyz),
                static_coeff_map_exyz=dict(static_coeff_map_exyz),
                drive_maps_exyz=drive_maps_exyz,
                a_row=[float(v) for v in a_row],
                s_static=float(s_static[k]),
                coeff_drop_abs_tol=float(coeff_drop_abs_tol),
            )
            qop = build_time_dependent_sparse_qop(
                ordered_labels_exyz=ordered_labels_exyz,
                static_coeff_map_exyz=static_coeff_map_exyz,
                drive_coeff_map_exyz=stage_map,
            )
            qc.append(PauliEvolutionGate(qop, time=float(dt), synthesis=synthesis), qubits)
    return qc


def _pauli_weight(label_exyz: str) -> int:
    return int(sum(1 for ch in str(label_exyz) if ch in {"x", "y", "z"}))


def _pauli_xy_count(label_exyz: str) -> int:
    return int(sum(1 for ch in str(label_exyz) if ch in {"x", "y"}))


def _cx_proxy_term(label_exyz: str) -> int:
    return int(2 * max(_pauli_weight(label_exyz) - 1, 0))


def _sq_proxy_term(label_exyz: str) -> int:
    return int(2 * _pauli_xy_count(label_exyz) + 1)


def _active_labels_exyz(
    coeff_map_exyz: dict[str, complex],
    ordered_labels_exyz: list[str],
    tol: float,
) -> list[str]:
    thr = float(max(0.0, tol))
    out: list[str] = []
    for lbl in ordered_labels_exyz:
        if abs(complex(coeff_map_exyz.get(lbl, 0.0 + 0.0j))) > thr:
            out.append(lbl)
    return out


def _compute_sweep_proxy_cost(active_labels_exyz: list[str]) -> dict[str, int]:
    term_exp_count = int(2 * len(active_labels_exyz))
    cx_proxy = int(2 * sum(_cx_proxy_term(lbl) for lbl in active_labels_exyz))
    sq_proxy = int(2 * sum(_sq_proxy_term(lbl) for lbl in active_labels_exyz))
    return {
        "term_exp_count": int(term_exp_count),
        "cx_proxy": int(cx_proxy),
        "sq_proxy": int(sq_proxy),
    }


def _compute_time_dynamics_proxy_cost(
    *,
    method: str,
    t_final: float,
    trotter_steps: int,
    drive_t0: float,
    drive_time_sampling: str,
    ordered_labels_exyz: list[str],
    static_coeff_map_exyz: dict[str, complex],
    drive_provider_exyz: Any | None,
    active_coeff_tol: float,
    coeff_drop_abs_tol: float,
) -> dict[str, int]:
    method_norm = str(method).strip().lower()
    if method_norm not in _NOISY_METHODS_ALLOWED:
        raise ValueError(f"Unsupported noisy method {method!r}.")
    if int(trotter_steps) < 1:
        raise ValueError("trotter_steps must be >= 1")
    if float(t_final) < 0.0:
        raise ValueError("t_final must be >= 0")

    total_term = 0
    total_cx = 0
    total_sq = 0
    dt = float(t_final) / float(trotter_steps)

    if method_norm == "suzuki2":
        for step_idx in range(int(trotter_steps)):
            t_sample = _time_sample(step_idx, dt, str(drive_time_sampling))
            raw = {} if drive_provider_exyz is None else dict(drive_provider_exyz(float(drive_t0) + float(t_sample)))
            merged: dict[str, complex] = {}
            for lbl in ordered_labels_exyz:
                merged[lbl] = complex(static_coeff_map_exyz.get(lbl, 0.0 + 0.0j)) + complex(raw.get(lbl, 0.0))
            active = _active_labels_exyz(merged, ordered_labels_exyz, float(active_coeff_tol))
            sweep = _compute_sweep_proxy_cost(active)
            total_term += int(sweep["term_exp_count"])
            total_cx += int(sweep["cx_proxy"])
            total_sq += int(sweep["sq_proxy"])
    else:
        scheme = get_cfqm_scheme(str(method_norm))
        c_nodes = [float(x) for x in scheme["c"]]
        a_rows = [[float(v) for v in row] for row in scheme["a"]]
        s_static = [float(v) for v in scheme["s_static"]]

        for step_idx in range(int(trotter_steps)):
            t_abs = float(drive_t0) + float(step_idx) * float(dt)
            drive_maps_exyz: list[dict[str, complex]] = []
            for c_j in c_nodes:
                t_node = float(t_abs) + float(c_j) * float(dt)
                raw = {} if drive_provider_exyz is None else dict(drive_provider_exyz(float(t_node)))
                drive_maps_exyz.append({str(k): complex(v) for k, v in raw.items()})

            for k, a_row in enumerate(a_rows):
                stage_map = _build_cfqm_stage_map_exyz(
                    ordered_labels_exyz=list(ordered_labels_exyz),
                    static_coeff_map_exyz=dict(static_coeff_map_exyz),
                    drive_maps_exyz=drive_maps_exyz,
                    a_row=[float(v) for v in a_row],
                    s_static=float(s_static[k]),
                    coeff_drop_abs_tol=float(coeff_drop_abs_tol),
                )
                active = _active_labels_exyz(stage_map, ordered_labels_exyz, float(active_coeff_tol))
                sweep = _compute_sweep_proxy_cost(active)
                total_term += int(sweep["term_exp_count"])
                total_cx += int(sweep["cx_proxy"])
                total_sq += int(sweep["sq_proxy"])

    return {
        "term_exp_count_total": int(total_term),
        "pauli_rot_count_total": int(total_term),
        "cx_proxy_total": int(total_cx),
        "sq_proxy_total": int(total_sq),
        "depth_proxy_total": int(total_term),
    }


def _run_noisy_suzuki_trajectory(
    *,
    L: int,
    ordering: str,
    psi_seed: np.ndarray,
    ordered_labels_exyz: list[str],
    static_coeff_map_exyz: dict[str, complex],
    t_final: float,
    num_times: int,
    trotter_steps: int,
    drive_profile: dict[str, Any] | None,
    noise_mode: str,
    shots: int,
    seed: int,
    oracle_repeats: int,
    oracle_aggregate: str,
    mitigation_config: dict[str, Any],
    symmetry_mitigation_config: dict[str, Any],
    backend_name: str | None,
    use_fake_backend: bool,
    allow_aer_fallback: bool,
    omp_shm_workaround: bool,
) -> dict[str, Any]:
    nq = int(round(math.log2(int(np.asarray(psi_seed).size))))
    drive_provider_exyz, drive_meta = _drive_provider_from_profile(
        profile=drive_profile,
        num_sites=int(L),
        nq_total=int(nq),
        ordering=str(ordering),
    )

    initial_circuit = QuantumCircuit(int(nq))
    _append_reference_state(initial_circuit, np.asarray(psi_seed, dtype=complex))

    static_qop = build_time_dependent_sparse_qop(
        ordered_labels_exyz=ordered_labels_exyz,
        static_coeff_map_exyz=static_coeff_map_exyz,
        drive_coeff_map_exyz=None,
    )

    up0_idx = _spin_orbital_bit_index(0, 0, int(L), ordering)
    dn0_idx = _spin_orbital_bit_index(0, 1, int(L), ordering)
    obs_up0 = _number_operator_qop(int(nq), int(up0_idx))
    obs_dn0 = _number_operator_qop(int(nq), int(dn0_idx))
    obs_doublon0 = _doublon_site_qop(int(nq), int(up0_idx), int(dn0_idx))
    obs_staggered = _staggered_qop(int(nq), int(L), str(ordering))

    ideal_symmetry_mitigation_config = normalize_ideal_reference_symmetry_mitigation(
        symmetry_mitigation_config,
        noise_mode=str(noise_mode),
    )

    noisy_cfg = OracleConfig(
        noise_mode=str(noise_mode),
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        backend_name=(None if backend_name is None else str(backend_name)),
        use_fake_backend=bool(use_fake_backend),
        allow_aer_fallback=bool(allow_aer_fallback),
        aer_fallback_mode="sampler_shots",
        omp_shm_workaround=bool(omp_shm_workaround),
        mitigation=dict(mitigation_config),
        symmetry_mitigation=dict(symmetry_mitigation_config),
    )
    ideal_cfg = OracleConfig(
        noise_mode="ideal",
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        backend_name=None,
        use_fake_backend=False,
        mitigation={"mode": "none", "zne_scales": [], "dd_sequence": None},
        symmetry_mitigation=dict(ideal_symmetry_mitigation_config),
    )

    times = np.linspace(0.0, float(t_final), int(num_times))
    rows: list[dict[str, Any]] = []

    with ExpectationOracle(noisy_cfg) as noisy_oracle, ExpectationOracle(ideal_cfg) as ideal_oracle:
        for t_val in times:
            qc_t = _build_suzuki2_time_dependent_circuit(
                initial_circuit=initial_circuit,
                ordered_labels_exyz=ordered_labels_exyz,
                static_coeff_map_exyz=static_coeff_map_exyz,
                drive_provider_exyz=drive_provider_exyz,
                time_value=float(t_val),
                trotter_steps=int(trotter_steps),
                drive_t0=float(0.0 if drive_profile is None else drive_profile.get("t0", 0.0)),
                drive_time_sampling=str(
                    "midpoint" if drive_profile is None else drive_profile.get("time_sampling", "midpoint")
                ),
            )

            if drive_provider_exyz is None:
                total_qop = static_qop
            else:
                drive_obs_map = dict(
                    drive_provider_exyz(
                        float(drive_profile.get("t0", 0.0)) + float(t_val)
                    )
                )
                total_qop = build_time_dependent_sparse_qop(
                    ordered_labels_exyz=ordered_labels_exyz,
                    static_coeff_map_exyz=static_coeff_map_exyz,
                    drive_coeff_map_exyz=drive_obs_map,
                )

            def _pair(obs: SparsePauliOp) -> dict[str, float]:
                n_est = noisy_oracle.evaluate(qc_t, obs)
                i_est = ideal_oracle.evaluate(qc_t, obs)
                delta_mean = float(n_est.mean - i_est.mean)
                delta_stderr = _combine_stderr(n_est.stderr, i_est.stderr)
                return {
                    "noisy_mean": float(n_est.mean),
                    "noisy_std": float(n_est.std),
                    "noisy_stdev": float(n_est.stdev),
                    "noisy_stderr": float(n_est.stderr),
                    "ideal_mean": float(i_est.mean),
                    "ideal_std": float(i_est.std),
                    "ideal_stdev": float(i_est.stdev),
                    "ideal_stderr": float(i_est.stderr),
                    "delta_mean": float(delta_mean),
                    "delta_stderr": float(delta_stderr),
                }

            e_s = _pair(static_qop)
            e_t = _pair(total_qop)
            up0 = _pair(obs_up0)
            dn0 = _pair(obs_dn0)
            dbl = _pair(obs_doublon0)
            stg = _pair(obs_staggered)

            rows.append(
                {
                    "time": float(t_val),
                    "energy_static_noisy": e_s["noisy_mean"],
                    "energy_static_noisy_mean": e_s["noisy_mean"],
                    "energy_static_noisy_std": e_s["noisy_std"],
                    "energy_static_noisy_stdev": e_s["noisy_stdev"],
                    "energy_static_noisy_stderr": e_s["noisy_stderr"],
                    "energy_static_ideal": e_s["ideal_mean"],
                    "energy_static_ideal_mean": e_s["ideal_mean"],
                    "energy_static_ideal_std": e_s["ideal_std"],
                    "energy_static_ideal_stdev": e_s["ideal_stdev"],
                    "energy_static_ideal_stderr": e_s["ideal_stderr"],
                    "energy_static_delta_noisy_minus_ideal": e_s["delta_mean"],
                    "energy_static_delta_noisy_minus_ideal_mean": e_s["delta_mean"],
                    "energy_static_delta_noisy_minus_ideal_stderr": e_s["delta_stderr"],
                    "energy_total_noisy": e_t["noisy_mean"],
                    "energy_total_noisy_mean": e_t["noisy_mean"],
                    "energy_total_noisy_std": e_t["noisy_std"],
                    "energy_total_noisy_stdev": e_t["noisy_stdev"],
                    "energy_total_noisy_stderr": e_t["noisy_stderr"],
                    "energy_total_ideal": e_t["ideal_mean"],
                    "energy_total_ideal_mean": e_t["ideal_mean"],
                    "energy_total_ideal_std": e_t["ideal_std"],
                    "energy_total_ideal_stdev": e_t["ideal_stdev"],
                    "energy_total_ideal_stderr": e_t["ideal_stderr"],
                    "energy_total_delta_noisy_minus_ideal": e_t["delta_mean"],
                    "energy_total_delta_noisy_minus_ideal_mean": e_t["delta_mean"],
                    "energy_total_delta_noisy_minus_ideal_stderr": e_t["delta_stderr"],
                    "n_up_site0_noisy": up0["noisy_mean"],
                    "n_up_site0_noisy_mean": up0["noisy_mean"],
                    "n_up_site0_noisy_std": up0["noisy_std"],
                    "n_up_site0_noisy_stdev": up0["noisy_stdev"],
                    "n_up_site0_noisy_stderr": up0["noisy_stderr"],
                    "n_up_site0_ideal": up0["ideal_mean"],
                    "n_up_site0_ideal_mean": up0["ideal_mean"],
                    "n_up_site0_ideal_std": up0["ideal_std"],
                    "n_up_site0_ideal_stdev": up0["ideal_stdev"],
                    "n_up_site0_ideal_stderr": up0["ideal_stderr"],
                    "n_up_site0_delta_noisy_minus_ideal": up0["delta_mean"],
                    "n_up_site0_delta_noisy_minus_ideal_mean": up0["delta_mean"],
                    "n_up_site0_delta_noisy_minus_ideal_stderr": up0["delta_stderr"],
                    "n_dn_site0_noisy": dn0["noisy_mean"],
                    "n_dn_site0_noisy_mean": dn0["noisy_mean"],
                    "n_dn_site0_noisy_std": dn0["noisy_std"],
                    "n_dn_site0_noisy_stdev": dn0["noisy_stdev"],
                    "n_dn_site0_noisy_stderr": dn0["noisy_stderr"],
                    "n_dn_site0_ideal": dn0["ideal_mean"],
                    "n_dn_site0_ideal_mean": dn0["ideal_mean"],
                    "n_dn_site0_ideal_std": dn0["ideal_std"],
                    "n_dn_site0_ideal_stdev": dn0["ideal_stdev"],
                    "n_dn_site0_ideal_stderr": dn0["ideal_stderr"],
                    "n_dn_site0_delta_noisy_minus_ideal": dn0["delta_mean"],
                    "n_dn_site0_delta_noisy_minus_ideal_mean": dn0["delta_mean"],
                    "n_dn_site0_delta_noisy_minus_ideal_stderr": dn0["delta_stderr"],
                    "doublon_noisy": dbl["noisy_mean"],
                    "doublon_noisy_mean": dbl["noisy_mean"],
                    "doublon_noisy_std": dbl["noisy_std"],
                    "doublon_noisy_stdev": dbl["noisy_stdev"],
                    "doublon_noisy_stderr": dbl["noisy_stderr"],
                    "doublon_ideal": dbl["ideal_mean"],
                    "doublon_ideal_mean": dbl["ideal_mean"],
                    "doublon_ideal_std": dbl["ideal_std"],
                    "doublon_ideal_stdev": dbl["ideal_stdev"],
                    "doublon_ideal_stderr": dbl["ideal_stderr"],
                    "doublon_delta_noisy_minus_ideal": dbl["delta_mean"],
                    "doublon_delta_noisy_minus_ideal_mean": dbl["delta_mean"],
                    "doublon_delta_noisy_minus_ideal_stderr": dbl["delta_stderr"],
                    "staggered_noisy": stg["noisy_mean"],
                    "staggered_noisy_mean": stg["noisy_mean"],
                    "staggered_noisy_std": stg["noisy_std"],
                    "staggered_noisy_stdev": stg["noisy_stdev"],
                    "staggered_noisy_stderr": stg["noisy_stderr"],
                    "staggered_ideal": stg["ideal_mean"],
                    "staggered_ideal_mean": stg["ideal_mean"],
                    "staggered_ideal_std": stg["ideal_std"],
                    "staggered_ideal_stdev": stg["ideal_stdev"],
                    "staggered_ideal_stderr": stg["ideal_stderr"],
                    "staggered_delta_noisy_minus_ideal": stg["delta_mean"],
                    "staggered_delta_noisy_minus_ideal_mean": stg["delta_mean"],
                    "staggered_delta_noisy_minus_ideal_stderr": stg["delta_stderr"],
                }
            )

        backend_details = {
            "backend_info": {
                "noise_mode": str(noisy_oracle.backend_info.noise_mode),
                "estimator_kind": str(noisy_oracle.backend_info.estimator_kind),
                "backend_name": noisy_oracle.backend_info.backend_name,
                "using_fake_backend": bool(noisy_oracle.backend_info.using_fake_backend),
                "details": dict(noisy_oracle.backend_info.details),
            }
        }

    return {
        "success": True,
        "noise_mode": str(noise_mode),
        "drive_enabled": bool(drive_profile is not None),
        "drive_meta": drive_meta,
        "noise_config": {
            "noise_mode": str(noise_mode),
            "shots": int(shots),
            "oracle_repeats": int(oracle_repeats),
            "oracle_aggregate": str(oracle_aggregate),
            "mitigation": dict(mitigation_config),
            "symmetry_mitigation": dict(symmetry_mitigation_config),
        },
        "trajectory": rows,
        "delta_uncertainty": _trajectory_delta_uncertainty(rows),
        **backend_details,
    }


def _run_noisy_method_trajectory(
    *,
    L: int,
    ordering: str,
    psi_seed: np.ndarray,
    ordered_labels_exyz: list[str],
    static_coeff_map_exyz: dict[str, complex],
    t_final: float,
    num_times: int,
    trotter_steps: int,
    drive_profile: dict[str, Any] | None,
    noise_mode: str,
    shots: int,
    seed: int,
    oracle_repeats: int,
    oracle_aggregate: str,
    mitigation_config: dict[str, Any],
    symmetry_mitigation_config: dict[str, Any],
    backend_name: str | None,
    use_fake_backend: bool,
    allow_aer_fallback: bool,
    omp_shm_workaround: bool,
    method: str,
    benchmark_active_coeff_tol: float,
    cfqm_coeff_drop_abs_tol: float,
) -> dict[str, Any]:
    t_wall_start = float(time.perf_counter())
    method_norm = str(method).strip().lower()
    if method_norm not in _NOISY_METHODS_ALLOWED:
        raise ValueError(f"Unsupported noisy method {method!r}.")

    nq = int(round(math.log2(int(np.asarray(psi_seed).size))))
    drive_provider_exyz, drive_meta = _drive_provider_from_profile(
        profile=drive_profile,
        num_sites=int(L),
        nq_total=int(nq),
        ordering=str(ordering),
    )

    initial_circuit = QuantumCircuit(int(nq))
    _append_reference_state(initial_circuit, np.asarray(psi_seed, dtype=complex))

    static_qop = build_time_dependent_sparse_qop(
        ordered_labels_exyz=ordered_labels_exyz,
        static_coeff_map_exyz=static_coeff_map_exyz,
        drive_coeff_map_exyz=None,
    )

    up0_idx = _spin_orbital_bit_index(0, 0, int(L), ordering)
    dn0_idx = _spin_orbital_bit_index(0, 1, int(L), ordering)
    obs_up0 = _number_operator_qop(int(nq), int(up0_idx))
    obs_dn0 = _number_operator_qop(int(nq), int(dn0_idx))
    obs_doublon0 = _doublon_site_qop(int(nq), int(up0_idx), int(dn0_idx))
    obs_staggered = _staggered_qop(int(nq), int(L), str(ordering))

    ideal_symmetry_mitigation_config = normalize_ideal_reference_symmetry_mitigation(
        symmetry_mitigation_config,
        noise_mode=str(noise_mode),
    )

    noisy_cfg = OracleConfig(
        noise_mode=str(noise_mode),
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        backend_name=(None if backend_name is None else str(backend_name)),
        use_fake_backend=bool(use_fake_backend),
        allow_aer_fallback=bool(allow_aer_fallback),
        aer_fallback_mode="sampler_shots",
        omp_shm_workaround=bool(omp_shm_workaround),
        mitigation=dict(mitigation_config),
        symmetry_mitigation=dict(symmetry_mitigation_config),
    )
    ideal_cfg = OracleConfig(
        noise_mode="ideal",
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        backend_name=None,
        use_fake_backend=False,
        mitigation={"mode": "none", "zne_scales": [], "dd_sequence": None},
        symmetry_mitigation=dict(ideal_symmetry_mitigation_config),
    )

    times = np.linspace(0.0, float(t_final), int(num_times))
    rows: list[dict[str, Any]] = []
    circuit_build_s_total = 0.0
    oracle_eval_s_total = 0.0
    oracle_calls_total = 0

    with ExpectationOracle(noisy_cfg) as noisy_oracle, ExpectationOracle(ideal_cfg) as ideal_oracle:
        for t_val in times:
            t_circ0 = float(time.perf_counter())
            if method_norm == "suzuki2":
                qc_t = _build_suzuki2_time_dependent_circuit(
                    initial_circuit=initial_circuit,
                    ordered_labels_exyz=ordered_labels_exyz,
                    static_coeff_map_exyz=static_coeff_map_exyz,
                    drive_provider_exyz=drive_provider_exyz,
                    time_value=float(t_val),
                    trotter_steps=int(trotter_steps),
                    drive_t0=float(0.0 if drive_profile is None else drive_profile.get("t0", 0.0)),
                    drive_time_sampling=str(
                        "midpoint" if drive_profile is None else drive_profile.get("time_sampling", "midpoint")
                    ),
                )
            else:
                qc_t = _build_cfqm_time_dependent_circuit(
                    method=str(method_norm),
                    initial_circuit=initial_circuit,
                    ordered_labels_exyz=ordered_labels_exyz,
                    static_coeff_map_exyz=static_coeff_map_exyz,
                    drive_provider_exyz=drive_provider_exyz,
                    time_value=float(t_val),
                    trotter_steps=int(trotter_steps),
                    drive_t0=float(0.0 if drive_profile is None else drive_profile.get("t0", 0.0)),
                    coeff_drop_abs_tol=float(cfqm_coeff_drop_abs_tol),
                )
            circuit_build_s_total += float(time.perf_counter() - t_circ0)

            if drive_provider_exyz is None:
                total_qop = static_qop
            else:
                drive_obs_map = dict(
                    drive_provider_exyz(
                        float(drive_profile.get("t0", 0.0)) + float(t_val)
                    )
                )
                total_qop = build_time_dependent_sparse_qop(
                    ordered_labels_exyz=ordered_labels_exyz,
                    static_coeff_map_exyz=static_coeff_map_exyz,
                    drive_coeff_map_exyz=drive_obs_map,
                )

            def _pair(obs: SparsePauliOp) -> dict[str, float]:
                nonlocal oracle_eval_s_total, oracle_calls_total
                t_eval0 = float(time.perf_counter())
                n_est = noisy_oracle.evaluate(qc_t, obs)
                i_est = ideal_oracle.evaluate(qc_t, obs)
                oracle_eval_s_total += float(time.perf_counter() - t_eval0)
                oracle_calls_total += 2
                delta_mean = float(n_est.mean - i_est.mean)
                delta_stderr = _combine_stderr(n_est.stderr, i_est.stderr)
                return {
                    "noisy_mean": float(n_est.mean),
                    "noisy_std": float(n_est.std),
                    "noisy_stdev": float(n_est.stdev),
                    "noisy_stderr": float(n_est.stderr),
                    "ideal_mean": float(i_est.mean),
                    "ideal_std": float(i_est.std),
                    "ideal_stdev": float(i_est.stdev),
                    "ideal_stderr": float(i_est.stderr),
                    "delta_mean": float(delta_mean),
                    "delta_stderr": float(delta_stderr),
                }

            e_s = _pair(static_qop)
            e_t = _pair(total_qop)
            up0 = _pair(obs_up0)
            dn0 = _pair(obs_dn0)
            dbl = _pair(obs_doublon0)
            stg = _pair(obs_staggered)

            rows.append(
                {
                    "time": float(t_val),
                    "energy_static_noisy": e_s["noisy_mean"],
                    "energy_static_noisy_mean": e_s["noisy_mean"],
                    "energy_static_noisy_std": e_s["noisy_std"],
                    "energy_static_noisy_stdev": e_s["noisy_stdev"],
                    "energy_static_noisy_stderr": e_s["noisy_stderr"],
                    "energy_static_ideal": e_s["ideal_mean"],
                    "energy_static_ideal_mean": e_s["ideal_mean"],
                    "energy_static_ideal_std": e_s["ideal_std"],
                    "energy_static_ideal_stdev": e_s["ideal_stdev"],
                    "energy_static_ideal_stderr": e_s["ideal_stderr"],
                    "energy_static_delta_noisy_minus_ideal": e_s["delta_mean"],
                    "energy_static_delta_noisy_minus_ideal_mean": e_s["delta_mean"],
                    "energy_static_delta_noisy_minus_ideal_stderr": e_s["delta_stderr"],
                    "energy_total_noisy": e_t["noisy_mean"],
                    "energy_total_noisy_mean": e_t["noisy_mean"],
                    "energy_total_noisy_std": e_t["noisy_std"],
                    "energy_total_noisy_stdev": e_t["noisy_stdev"],
                    "energy_total_noisy_stderr": e_t["noisy_stderr"],
                    "energy_total_ideal": e_t["ideal_mean"],
                    "energy_total_ideal_mean": e_t["ideal_mean"],
                    "energy_total_ideal_std": e_t["ideal_std"],
                    "energy_total_ideal_stdev": e_t["ideal_stdev"],
                    "energy_total_ideal_stderr": e_t["ideal_stderr"],
                    "energy_total_delta_noisy_minus_ideal": e_t["delta_mean"],
                    "energy_total_delta_noisy_minus_ideal_mean": e_t["delta_mean"],
                    "energy_total_delta_noisy_minus_ideal_stderr": e_t["delta_stderr"],
                    "n_up_site0_noisy": up0["noisy_mean"],
                    "n_up_site0_noisy_mean": up0["noisy_mean"],
                    "n_up_site0_noisy_std": up0["noisy_std"],
                    "n_up_site0_noisy_stdev": up0["noisy_stdev"],
                    "n_up_site0_noisy_stderr": up0["noisy_stderr"],
                    "n_up_site0_ideal": up0["ideal_mean"],
                    "n_up_site0_ideal_mean": up0["ideal_mean"],
                    "n_up_site0_ideal_std": up0["ideal_std"],
                    "n_up_site0_ideal_stdev": up0["ideal_stdev"],
                    "n_up_site0_ideal_stderr": up0["ideal_stderr"],
                    "n_up_site0_delta_noisy_minus_ideal": up0["delta_mean"],
                    "n_up_site0_delta_noisy_minus_ideal_mean": up0["delta_mean"],
                    "n_up_site0_delta_noisy_minus_ideal_stderr": up0["delta_stderr"],
                    "n_dn_site0_noisy": dn0["noisy_mean"],
                    "n_dn_site0_noisy_mean": dn0["noisy_mean"],
                    "n_dn_site0_noisy_std": dn0["noisy_std"],
                    "n_dn_site0_noisy_stdev": dn0["noisy_stdev"],
                    "n_dn_site0_noisy_stderr": dn0["noisy_stderr"],
                    "n_dn_site0_ideal": dn0["ideal_mean"],
                    "n_dn_site0_ideal_mean": dn0["ideal_mean"],
                    "n_dn_site0_ideal_std": dn0["ideal_std"],
                    "n_dn_site0_ideal_stdev": dn0["ideal_stdev"],
                    "n_dn_site0_ideal_stderr": dn0["ideal_stderr"],
                    "n_dn_site0_delta_noisy_minus_ideal": dn0["delta_mean"],
                    "n_dn_site0_delta_noisy_minus_ideal_mean": dn0["delta_mean"],
                    "n_dn_site0_delta_noisy_minus_ideal_stderr": dn0["delta_stderr"],
                    "doublon_noisy": dbl["noisy_mean"],
                    "doublon_noisy_mean": dbl["noisy_mean"],
                    "doublon_noisy_std": dbl["noisy_std"],
                    "doublon_noisy_stdev": dbl["noisy_stdev"],
                    "doublon_noisy_stderr": dbl["noisy_stderr"],
                    "doublon_ideal": dbl["ideal_mean"],
                    "doublon_ideal_mean": dbl["ideal_mean"],
                    "doublon_ideal_std": dbl["ideal_std"],
                    "doublon_ideal_stdev": dbl["ideal_stdev"],
                    "doublon_ideal_stderr": dbl["ideal_stderr"],
                    "doublon_delta_noisy_minus_ideal": dbl["delta_mean"],
                    "doublon_delta_noisy_minus_ideal_mean": dbl["delta_mean"],
                    "doublon_delta_noisy_minus_ideal_stderr": dbl["delta_stderr"],
                    "staggered_noisy": stg["noisy_mean"],
                    "staggered_noisy_mean": stg["noisy_mean"],
                    "staggered_noisy_std": stg["noisy_std"],
                    "staggered_noisy_stdev": stg["noisy_stdev"],
                    "staggered_noisy_stderr": stg["noisy_stderr"],
                    "staggered_ideal": stg["ideal_mean"],
                    "staggered_ideal_mean": stg["ideal_mean"],
                    "staggered_ideal_std": stg["ideal_std"],
                    "staggered_ideal_stdev": stg["ideal_stdev"],
                    "staggered_ideal_stderr": stg["ideal_stderr"],
                    "staggered_delta_noisy_minus_ideal": stg["delta_mean"],
                    "staggered_delta_noisy_minus_ideal_mean": stg["delta_mean"],
                    "staggered_delta_noisy_minus_ideal_stderr": stg["delta_stderr"],
                }
            )

        backend_details = {
            "backend_info": {
                "noise_mode": str(noisy_oracle.backend_info.noise_mode),
                "estimator_kind": str(noisy_oracle.backend_info.estimator_kind),
                "backend_name": noisy_oracle.backend_info.backend_name,
                "using_fake_backend": bool(noisy_oracle.backend_info.using_fake_backend),
                "details": dict(noisy_oracle.backend_info.details),
            }
        }

    benchmark_cost = _compute_time_dynamics_proxy_cost(
        method=str(method_norm),
        t_final=float(t_final),
        trotter_steps=int(trotter_steps),
        drive_t0=float(0.0 if drive_profile is None else drive_profile.get("t0", 0.0)),
        drive_time_sampling=str(
            "midpoint" if drive_profile is None else drive_profile.get("time_sampling", "midpoint")
        ),
        ordered_labels_exyz=list(ordered_labels_exyz),
        static_coeff_map_exyz=dict(static_coeff_map_exyz),
        drive_provider_exyz=drive_provider_exyz,
        active_coeff_tol=float(benchmark_active_coeff_tol),
        coeff_drop_abs_tol=float(cfqm_coeff_drop_abs_tol),
    )
    benchmark_runtime = {
        "wall_total_s": float(time.perf_counter() - t_wall_start),
        "circuit_build_s_total": float(circuit_build_s_total),
        "oracle_eval_s_total": float(oracle_eval_s_total),
        "oracle_calls_total": int(oracle_calls_total),
        "trajectory_points": int(len(rows)),
    }

    return {
        "success": True,
        "method": str(method_norm),
        "noise_mode": str(noise_mode),
        "drive_enabled": bool(drive_profile is not None),
        "drive_meta": drive_meta,
        "noise_config": {
            "noise_mode": str(noise_mode),
            "shots": int(shots),
            "oracle_repeats": int(oracle_repeats),
            "oracle_aggregate": str(oracle_aggregate),
            "mitigation": dict(mitigation_config),
            "symmetry_mitigation": dict(symmetry_mitigation_config),
        },
        "trajectory": rows,
        "delta_uncertainty": _trajectory_delta_uncertainty(rows),
        "benchmark_cost": benchmark_cost,
        "benchmark_runtime": benchmark_runtime,
        **backend_details,
    }


def _run_static_observable_audit_core(
    *,
    L: int,
    ordering: str,
    initial_circuit: QuantumCircuit,
    ordered_labels_exyz: list[str],
    static_coeff_map_exyz: dict[str, complex],
    drive_profile: dict[str, Any] | None,
    noise_mode: str,
    shots: int,
    seed: int,
    oracle_repeats: int,
    oracle_aggregate: str,
    mitigation_config: dict[str, Any],
    symmetry_mitigation_config: dict[str, Any],
    backend_name: str | None,
    use_fake_backend: bool,
    allow_aer_fallback: bool,
    omp_shm_workaround: bool,
    audit_source: dict[str, Any] | None = None,
    extra_meta: dict[str, Any] | None = None,
    seed_transpiler: int | None = None,
    transpile_optimization_level: int | None = None,
    compile_request_source: str | None = None,
) -> dict[str, Any]:
    nq = int(initial_circuit.num_qubits)
    drive_provider_exyz, drive_meta = _drive_provider_from_profile(
        profile=drive_profile,
        num_sites=int(L),
        nq_total=int(nq),
        ordering=str(ordering),
    )

    static_qop = build_time_dependent_sparse_qop(
        ordered_labels_exyz=ordered_labels_exyz,
        static_coeff_map_exyz=static_coeff_map_exyz,
        drive_coeff_map_exyz=None,
    )
    if drive_provider_exyz is None:
        total_qop = static_qop
    else:
        t0 = float(0.0 if drive_profile is None else drive_profile.get("t0", 0.0))
        drive_obs_map = dict(drive_provider_exyz(t0))
        total_qop = build_time_dependent_sparse_qop(
            ordered_labels_exyz=ordered_labels_exyz,
            static_coeff_map_exyz=static_coeff_map_exyz,
            drive_coeff_map_exyz=drive_obs_map,
        )

    up0_idx = _spin_orbital_bit_index(0, 0, int(L), ordering)
    dn0_idx = _spin_orbital_bit_index(0, 1, int(L), ordering)
    obs_up0 = _number_operator_qop(int(nq), int(up0_idx))
    obs_dn0 = _number_operator_qop(int(nq), int(dn0_idx))
    obs_doublon0 = _doublon_site_qop(int(nq), int(up0_idx), int(dn0_idx))
    obs_staggered = _staggered_qop(int(nq), int(L), str(ordering))

    ideal_symmetry_mitigation_config = normalize_ideal_reference_symmetry_mitigation(
        symmetry_mitigation_config,
        noise_mode=str(noise_mode),
    )
    noisy_mitigation_config = (
        {"mode": "none", "zne_scales": [], "dd_sequence": None, "local_readout_strategy": None}
        if str(noise_mode).strip().lower() == "ideal"
        else dict(normalize_mitigation_config(mitigation_config))
    )

    noisy_cfg = OracleConfig(
        noise_mode=str(noise_mode),
        shots=int(shots),
        seed=int(seed),
        seed_transpiler=(
            None if seed_transpiler is None else int(seed_transpiler)
        ),
        transpile_optimization_level=int(
            1
            if transpile_optimization_level is None
            else transpile_optimization_level
        ),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        backend_name=(None if backend_name is None else str(backend_name)),
        use_fake_backend=bool(use_fake_backend),
        allow_aer_fallback=bool(allow_aer_fallback),
        aer_fallback_mode="sampler_shots",
        omp_shm_workaround=bool(omp_shm_workaround),
        mitigation=dict(noisy_mitigation_config),
        symmetry_mitigation=dict(symmetry_mitigation_config),
    )
    ideal_cfg = OracleConfig(
        noise_mode="ideal",
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        backend_name=None,
        use_fake_backend=False,
        mitigation={"mode": "none", "zne_scales": [], "dd_sequence": None, "local_readout_strategy": None},
        symmetry_mitigation=dict(ideal_symmetry_mitigation_config),
    )

    with ExpectationOracle(noisy_cfg) as noisy_oracle, ExpectationOracle(ideal_cfg) as ideal_oracle:
        def _pair(obs: SparsePauliOp) -> dict[str, float]:
            n_est = noisy_oracle.evaluate(initial_circuit, obs)
            i_est = ideal_oracle.evaluate(initial_circuit, obs)
            delta_mean = float(n_est.mean - i_est.mean)
            delta_stderr = _combine_stderr(n_est.stderr, i_est.stderr)
            return {
                "noisy_mean": float(n_est.mean),
                "noisy_std": float(n_est.std),
                "noisy_stdev": float(n_est.stdev),
                "noisy_stderr": float(n_est.stderr),
                "ideal_mean": float(i_est.mean),
                "ideal_std": float(i_est.std),
                "ideal_stdev": float(i_est.stdev),
                "ideal_stderr": float(i_est.stderr),
                "delta_mean": float(delta_mean),
                "delta_stderr": float(delta_stderr),
            }

        obs_map = {
            "energy_static": _pair(static_qop),
            "energy_total": _pair(total_qop),
            "n_up_site0": _pair(obs_up0),
            "n_dn_site0": _pair(obs_dn0),
            "doublon": _pair(obs_doublon0),
            "staggered": _pair(obs_staggered),
        }

        backend_details = {
            "backend_info": {
                "noise_mode": str(noisy_oracle.backend_info.noise_mode),
                "estimator_kind": str(noisy_oracle.backend_info.estimator_kind),
                "backend_name": noisy_oracle.backend_info.backend_name,
                "using_fake_backend": bool(noisy_oracle.backend_info.using_fake_backend),
                "details": dict(noisy_oracle.backend_info.details),
            }
        }

    delta_unc: dict[str, dict[str, float]] = {}
    for name, rec in obs_map.items():
        delta_unc[name] = _delta_uncertainty_metrics(
            np.asarray([float(rec["delta_mean"])], dtype=float),
            np.asarray([float(rec["delta_stderr"])], dtype=float),
        )

    compile_control = _build_compile_request_payload(
        backend_name=(None if backend_name in {None, ""} else str(backend_name)),
        seed_transpiler=(None if seed_transpiler is None else int(seed_transpiler)),
        transpile_optimization_level=(
            None
            if transpile_optimization_level is None
            else int(transpile_optimization_level)
        ),
        source=(
            str(compile_request_source)
            if compile_request_source not in {None, ""}
            else "static_observable_audit"
        ),
    )
    compile_observation = _build_compile_observation_payload(
        requested=compile_control,
        backend_info=backend_details.get("backend_info", {}),
    )

    return {
        "success": True,
        "noise_mode": str(noise_mode),
        "drive_enabled": bool(drive_profile is not None),
        "drive_meta": drive_meta,
        "time": 0.0,
        "noise_config": {
            "noise_mode": str(noise_mode),
            "shots": int(shots),
            "oracle_repeats": int(oracle_repeats),
            "oracle_aggregate": str(oracle_aggregate),
            "mitigation": dict(noisy_mitigation_config),
            "symmetry_mitigation": dict(symmetry_mitigation_config),
        },
        "compile_control": dict(compile_control),
        "compile_observation": dict(compile_observation),
        "final_observables": obs_map,
        "delta_uncertainty": delta_unc,
        "audit_source": (dict(audit_source) if isinstance(audit_source, dict) else {}),
        "extra_meta": (dict(extra_meta) if isinstance(extra_meta, dict) else {}),
        **backend_details,
    }


def _run_noisy_final_state_audit(
    *,
    L: int,
    ordering: str,
    psi_seed: np.ndarray,
    ordered_labels_exyz: list[str],
    static_coeff_map_exyz: dict[str, complex],
    drive_profile: dict[str, Any] | None,
    noise_mode: str,
    shots: int,
    seed: int,
    oracle_repeats: int,
    oracle_aggregate: str,
    mitigation_config: dict[str, Any],
    symmetry_mitigation_config: dict[str, Any],
    backend_name: str | None,
    use_fake_backend: bool,
    allow_aer_fallback: bool,
    omp_shm_workaround: bool,
) -> dict[str, Any]:
    nq = int(round(math.log2(int(np.asarray(psi_seed).size))))
    initial_circuit = QuantumCircuit(int(nq))
    _append_reference_state(initial_circuit, np.asarray(psi_seed, dtype=complex))
    return _run_static_observable_audit_core(
        L=int(L),
        ordering=str(ordering),
        initial_circuit=initial_circuit,
        ordered_labels_exyz=list(ordered_labels_exyz),
        static_coeff_map_exyz=dict(static_coeff_map_exyz),
        drive_profile=drive_profile,
        noise_mode=str(noise_mode),
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        mitigation_config=dict(mitigation_config),
        symmetry_mitigation_config=dict(symmetry_mitigation_config),
        backend_name=backend_name,
        use_fake_backend=bool(use_fake_backend),
        allow_aer_fallback=bool(allow_aer_fallback),
        omp_shm_workaround=bool(omp_shm_workaround),
        audit_source={
            "kind": "replay_prepared_state",
            "includes_ansatz_stateprep_noise": False,
        },
    )


def _run_imported_prepared_state_audit(
    *,
    artifact_json: str,
    noise_mode: str,
    shots: int,
    seed: int,
    oracle_repeats: int,
    oracle_aggregate: str,
    mitigation_config: dict[str, Any],
    symmetry_mitigation_config: dict[str, Any],
    backend_name: str | None,
    use_fake_backend: bool,
    allow_aer_fallback: bool,
    omp_shm_workaround: bool,
) -> dict[str, Any]:
    ctx = _load_imported_artifact_context(artifact_json)
    prepared_state = ctx.get("prepared_state", None)
    if prepared_state is None:
        return {
            "success": False,
            "available": False,
            "reason": "initial_state_missing",
            "source_kind": str(ctx.get("source_kind", "unknown")),
            "artifact_json": str(ctx["path"]),
        }
    initial_circuit = QuantumCircuit(int(ctx["num_qubits"]))
    _append_reference_state(initial_circuit, np.asarray(prepared_state, dtype=complex).reshape(-1))
    return _run_static_observable_audit_core(
        L=int(ctx["settings"].get("L", 2)),
        ordering=str(ctx["settings"].get("ordering", "blocked")),
        initial_circuit=initial_circuit,
        ordered_labels_exyz=list(ctx["ordered_labels_exyz"]),
        static_coeff_map_exyz=dict(ctx["static_coeff_map_exyz"]),
        drive_profile=None,
        noise_mode=str(noise_mode),
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        mitigation_config=dict(mitigation_config),
        symmetry_mitigation_config=dict(symmetry_mitigation_config),
        backend_name=backend_name,
        use_fake_backend=bool(use_fake_backend),
        allow_aer_fallback=bool(allow_aer_fallback),
        omp_shm_workaround=bool(omp_shm_workaround),
        audit_source={
            "kind": "imported_prepared_state",
            "includes_ansatz_stateprep_noise": False,
            "source_kind": str(ctx.get("source_kind", "unknown")),
            "artifact_json": str(ctx["path"]),
        },
        extra_meta={
            "saved_artifact_energy": ctx.get("saved_energy", None),
            "exact_energy": ctx.get("exact_energy", None),
        },
    )


def _run_imported_ansatz_input_state_audit(
    *,
    artifact_json: str,
    noise_mode: str,
    shots: int,
    seed: int,
    oracle_repeats: int,
    oracle_aggregate: str,
    mitigation_config: dict[str, Any],
    symmetry_mitigation_config: dict[str, Any],
    backend_name: str | None,
    use_fake_backend: bool,
    allow_aer_fallback: bool,
    omp_shm_workaround: bool,
    seed_transpiler: int | None = None,
    transpile_optimization_level: int | None = None,
    compile_request_source: str | None = None,
) -> dict[str, Any]:
    ctx = _load_imported_artifact_context(artifact_json)
    ansatz_input_state_meta = dict(ctx.get("ansatz_input_state_meta", {}))
    if not bool(ansatz_input_state_meta.get("available", False)):
        return {
            "success": False,
            "available": False,
            "reason": str(
                ansatz_input_state_meta.get(
                    "reason",
                    "missing_ansatz_input_state_provenance",
                )
            ),
            "source_kind": str(ctx.get("source_kind", "unknown")),
            "artifact_json": str(ctx["path"]),
            "reference_state_embedded": False,
            "ansatz_input_state_source": ansatz_input_state_meta.get("source", None),
            "ansatz_input_state_kind": ansatz_input_state_meta.get(
                "handoff_state_kind",
                None,
            ),
            "error": ansatz_input_state_meta.get("error", None),
        }
    ansatz_input_state = ctx.get("ansatz_input_state", None)
    if ansatz_input_state is None:
        return {
            "success": False,
            "available": False,
            "reason": "ansatz_input_state_missing",
            "source_kind": str(ctx.get("source_kind", "unknown")),
            "artifact_json": str(ctx["path"]),
            "reference_state_embedded": False,
            "ansatz_input_state_source": ansatz_input_state_meta.get("source", None),
            "ansatz_input_state_kind": ansatz_input_state_meta.get(
                "handoff_state_kind",
                None,
            ),
            "error": None,
        }
    initial_circuit = QuantumCircuit(int(ctx["num_qubits"]))
    _append_reference_state(
        initial_circuit,
        np.asarray(ansatz_input_state, dtype=complex).reshape(-1),
    )
    payload = _run_static_observable_audit_core(
        L=int(ctx["settings"].get("L", 2)),
        ordering=str(ctx["settings"].get("ordering", "blocked")),
        initial_circuit=initial_circuit,
        ordered_labels_exyz=list(ctx["ordered_labels_exyz"]),
        static_coeff_map_exyz=dict(ctx["static_coeff_map_exyz"]),
        drive_profile=None,
        noise_mode=str(noise_mode),
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        mitigation_config=dict(mitigation_config),
        symmetry_mitigation_config=dict(symmetry_mitigation_config),
        backend_name=backend_name,
        use_fake_backend=bool(use_fake_backend),
        allow_aer_fallback=bool(allow_aer_fallback),
        omp_shm_workaround=bool(omp_shm_workaround),
        audit_source={
            "kind": "imported_ansatz_input_state",
            "includes_ansatz_stateprep_noise": False,
            "source_kind": str(ctx.get("source_kind", "unknown")),
            "artifact_json": str(ctx["path"]),
            "reference_state_embedded": True,
            "ansatz_input_state_source": ansatz_input_state_meta.get("source", None),
            "ansatz_input_state_kind": ansatz_input_state_meta.get(
                "handoff_state_kind",
                None,
            ),
        },
        extra_meta={
            "artifact_full_circuit_saved_energy": ctx.get("saved_energy", None),
            "exact_energy": ctx.get("exact_energy", None),
            "logical_parameter_count": int(ctx["layout"].logical_parameter_count),
            "runtime_parameter_count": int(ctx["layout"].runtime_parameter_count),
        },
        seed_transpiler=(None if seed_transpiler is None else int(seed_transpiler)),
        transpile_optimization_level=(
            None
            if transpile_optimization_level is None
            else int(transpile_optimization_level)
        ),
        compile_request_source=(
            None if compile_request_source in {None, ""} else str(compile_request_source)
        ),
    )
    energy_static = payload.get("final_observables", {}).get("energy_static", {})
    ideal_input_state_energy = (
        None
        if not isinstance(energy_static, dict)
        else energy_static.get("ideal_mean", None)
    )
    payload["reference_state_embedded"] = True
    payload["ansatz_input_state_source"] = ansatz_input_state_meta.get("source", None)
    payload["ansatz_input_state_kind"] = ansatz_input_state_meta.get(
        "handoff_state_kind",
        None,
    )
    payload["ansatz_input_state_reference"] = {
        "artifact_full_circuit_saved_energy": ctx.get("saved_energy", None),
        "ideal_input_state_energy": ideal_input_state_energy,
        "exact_energy": ctx.get("exact_energy", None),
        "logical_parameter_count": int(ctx["layout"].logical_parameter_count),
        "runtime_parameter_count": int(ctx["layout"].runtime_parameter_count),
    }
    return payload


def _run_imported_full_circuit_audit(
    *,
    artifact_json: str,
    noise_mode: str,
    shots: int,
    seed: int,
    oracle_repeats: int,
    oracle_aggregate: str,
    mitigation_config: dict[str, Any],
    symmetry_mitigation_config: dict[str, Any],
    backend_name: str | None,
    use_fake_backend: bool,
    allow_aer_fallback: bool,
    omp_shm_workaround: bool,
    seed_transpiler: int | None = None,
    transpile_optimization_level: int | None = None,
    compile_request_source: str | None = None,
) -> dict[str, Any]:
    ctx = _load_imported_artifact_context(artifact_json)
    ansatz_input_state_meta = dict(ctx.get("ansatz_input_state_meta", {}))
    if not bool(ansatz_input_state_meta.get("available", False)):
        return {
            "success": False,
            "available": False,
            "reason": str(ansatz_input_state_meta.get("reason", "missing_ansatz_input_state_provenance")),
            "source_kind": str(ctx.get("source_kind", "unknown")),
            "artifact_json": str(ctx["path"]),
            "reference_state_embedded": False,
            "ansatz_input_state_source": ansatz_input_state_meta.get("source", None),
            "ansatz_input_state_kind": ansatz_input_state_meta.get("handoff_state_kind", None),
            "error": ansatz_input_state_meta.get("error", None),
        }
    payload = _run_static_observable_audit_core(
        L=int(ctx["settings"].get("L", 2)),
        ordering=str(ctx["settings"].get("ordering", "blocked")),
        initial_circuit=ctx["circuit"],
        ordered_labels_exyz=list(ctx["ordered_labels_exyz"]),
        static_coeff_map_exyz=dict(ctx["static_coeff_map_exyz"]),
        drive_profile=None,
        noise_mode=str(noise_mode),
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        mitigation_config=dict(mitigation_config),
        symmetry_mitigation_config=dict(symmetry_mitigation_config),
        backend_name=backend_name,
        use_fake_backend=bool(use_fake_backend),
        allow_aer_fallback=bool(allow_aer_fallback),
        omp_shm_workaround=bool(omp_shm_workaround),
        audit_source={
            "kind": "full_circuit_import",
            "includes_ansatz_stateprep_noise": True,
            "source_kind": str(ctx.get("source_kind", "unknown")),
            "artifact_json": str(ctx["path"]),
            "reference_state_embedded": True,
            "ansatz_input_state_source": ansatz_input_state_meta.get("source", None),
            "ansatz_input_state_kind": ansatz_input_state_meta.get("handoff_state_kind", None),
        },
        extra_meta={
            "saved_artifact_energy": ctx.get("saved_energy", None),
            "exact_energy": ctx.get("exact_energy", None),
            "logical_parameter_count": int(ctx["layout"].logical_parameter_count),
            "runtime_parameter_count": int(ctx["layout"].runtime_parameter_count),
        },
        seed_transpiler=(None if seed_transpiler is None else int(seed_transpiler)),
        transpile_optimization_level=(
            None if transpile_optimization_level is None else int(transpile_optimization_level)
        ),
        compile_request_source=(
            None if compile_request_source in {None, ""} else str(compile_request_source)
        ),
    )
    energy_static = payload.get("final_observables", {}).get("energy_static", {})
    if isinstance(energy_static, dict):
        payload["full_circuit_reference"] = {
            "saved_artifact_energy": ctx.get("saved_energy", None),
            "ideal_circuit_energy": energy_static.get("ideal_mean", None),
            "ideal_circuit_minus_saved_artifact": (
                None
                if ctx.get("saved_energy", None) is None or energy_static.get("ideal_mean", None) is None
                else float(energy_static["ideal_mean"]) - float(ctx["saved_energy"])
            ),
        }
    return payload


def _run_imported_fixed_lean_noisy_replay(
    *,
    artifact_json: str,
    shots: int,
    seed: int,
    oracle_repeats: int,
    oracle_aggregate: str,
    backend_name: str | None,
    use_fake_backend: bool,
    allow_aer_fallback: bool,
    omp_shm_workaround: bool,
    optimizer_method: str,
    optimizer_seed: int,
    optimizer_maxiter: int,
    optimizer_wallclock_cap_s: int,
    spsa_a: float,
    spsa_c: float,
    spsa_alpha: float,
    spsa_gamma: float,
    spsa_A: float,
    spsa_avg_last: int,
    spsa_eval_repeats: int,
    spsa_eval_agg: str,
    mitigation_config: dict[str, Any],
    symmetry_mitigation_config: dict[str, Any],
) -> dict[str, Any]:
    ctx, ansatz_input_state_meta, pool_type, error_payload = _resolve_locked_imported_lean_context(
        artifact_json,
        nonlean_reason="fixed_lean_replay_requires_pareto_lean_l2_source",
    )
    if error_payload is not None or ctx is None:
        return dict(error_payload or {})

    mitigation_cfg = dict(normalize_mitigation_config(mitigation_config))
    symmetry_cfg = dict(normalize_symmetry_mitigation_config(symmetry_mitigation_config))
    if str(mitigation_cfg.get("mode", "none")) != "readout":
        return {
            "success": False,
            "available": False,
            "reason": "fixed_lean_replay_requires_readout_mitigation",
            "artifact_json": str(ctx["path"]),
        }
    strategy = str(mitigation_cfg.get("local_readout_strategy") or "mthree")
    if strategy != "mthree":
        return {
            "success": False,
            "available": False,
            "reason": "fixed_lean_replay_requires_mthree",
            "artifact_json": str(ctx["path"]),
        }
    mitigation_cfg["local_readout_strategy"] = "mthree"
    if str(symmetry_cfg.get("mode", "off")) != "off":
        return {
            "success": False,
            "available": False,
            "reason": "fixed_lean_replay_requires_symmetry_off",
            "artifact_json": str(ctx["path"]),
        }
    optimizer_method_key = str(optimizer_method).strip().lower()
    if optimizer_method_key not in {"spsa", "powell"}:
        return {
            "success": False,
            "available": False,
            "reason": "fixed_lean_replay_requires_spsa_or_powell",
            "artifact_json": str(ctx["path"]),
            "optimizer_method": str(optimizer_method),
        }

    layout = ctx["layout"]
    nq = int(ctx["num_qubits"])
    theta0 = np.asarray(ctx["theta_runtime"], dtype=float).reshape(-1)
    ref_state = np.asarray(ctx["ansatz_input_state"], dtype=complex).reshape(-1)
    ordered_labels_exyz = list(ctx["ordered_labels_exyz"])
    static_coeff_map_exyz = dict(ctx["static_coeff_map_exyz"])
    qop = build_time_dependent_sparse_qop(
        ordered_labels_exyz=ordered_labels_exyz,
        static_coeff_map_exyz=static_coeff_map_exyz,
        drive_coeff_map_exyz=None,
    )

    noisy_mode = "backend_scheduled" if bool(use_fake_backend) else "runtime"
    noisy_cfg = OracleConfig(
        noise_mode=str(noisy_mode),
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        backend_name=(None if backend_name is None else str(backend_name)),
        use_fake_backend=bool(use_fake_backend),
        allow_aer_fallback=bool(allow_aer_fallback),
        aer_fallback_mode="sampler_shots",
        omp_shm_workaround=bool(omp_shm_workaround),
        mitigation=dict(mitigation_cfg),
        symmetry_mitigation=dict(symmetry_cfg),
    )

    class _WallclockStop(RuntimeError):
        pass

    t0 = time.perf_counter()
    objective_history_tail: list[dict[str, Any]] = []
    objective_calls_total = 0
    best_eval: dict[str, Any] | None = None
    wallclock_hit = False
    optimizer_exception: Exception | None = None
    res = None
    seed_transpiler: int | None = None
    transpile_optimization_level = 1
    runtime_profile_cfg = normalize_runtime_estimator_profile_config(None)
    runtime_session_cfg = normalize_runtime_session_policy_config(None)

    def _emit_replay_log(*_args: Any, **_kwargs: Any) -> None:
        return

    with ExpectationOracle(noisy_cfg) as noisy_oracle:
        def _objective(theta: np.ndarray) -> float:
            nonlocal objective_calls_total, best_eval, wallclock_hit
            elapsed_s = float(time.perf_counter() - t0)
            if elapsed_s >= float(optimizer_wallclock_cap_s) and objective_calls_total > 0:
                wallclock_hit = True
                raise _WallclockStop("fixed lean noisy replay wallclock cap reached")
            theta_arr = np.asarray(theta, dtype=float).reshape(-1)
            qc = _build_ansatz_circuit(layout, theta_arr, int(nq), ref_state=ref_state)
            est = noisy_oracle.evaluate(qc, qop)
            objective_calls_total += 1
            row = {
                "call_index": int(objective_calls_total),
                "elapsed_s": float(time.perf_counter() - t0),
                "energy_noisy_mean": float(est.mean),
                "energy_noisy_stderr": float(est.stderr),
                "theta_runtime": [float(x) for x in theta_arr.tolist()],
            }
            _bounded_append(objective_history_tail, row)
            if (
                best_eval is None
                or float(est.mean) < float(best_eval["energy_noisy_mean"])
                or (
                    float(est.mean) == float(best_eval["energy_noisy_mean"])
                    and float(est.stderr) < float(best_eval["energy_noisy_stderr"])
                )
            ):
                best_eval = {
                    "energy_noisy_mean": float(est.mean),
                    "energy_noisy_stderr": float(est.stderr),
                    "theta_runtime": np.asarray(theta_arr, dtype=float),
                    "call_index": int(objective_calls_total),
                }
            return float(est.mean)

        try:
            if optimizer_method_key == "spsa":
                res = spsa_minimize(
                    fun=_objective,
                    x0=theta0,
                    maxiter=int(optimizer_maxiter),
                    seed=int(optimizer_seed),
                    a=float(spsa_a),
                    c=float(spsa_c),
                    alpha=float(spsa_alpha),
                    gamma=float(spsa_gamma),
                    A=float(spsa_A),
                    bounds=None,
                    project="none",
                    eval_repeats=int(spsa_eval_repeats),
                    eval_agg=str(spsa_eval_agg),
                    avg_last=int(spsa_avg_last),
                )
            else:
                try:
                    from scipy.optimize import minimize as scipy_minimize  # type: ignore
                except Exception as exc:
                    raise RuntimeError("SciPy minimize is unavailable for Powell fixed lean noisy replay.") from exc
                res = scipy_minimize(
                    _objective,
                    theta0,
                    method="Powell",
                    options={"maxiter": int(optimizer_maxiter)},
                )
        except _WallclockStop:
            wallclock_hit = True
        except Exception as exc:
            optimizer_exception = exc

        backend_info = {
            "noise_mode": str(noisy_oracle.backend_info.noise_mode),
            "estimator_kind": str(noisy_oracle.backend_info.estimator_kind),
            "backend_name": noisy_oracle.backend_info.backend_name,
            "using_fake_backend": bool(noisy_oracle.backend_info.using_fake_backend),
            "details": dict(noisy_oracle.backend_info.details),
        }

    if optimizer_exception is not None:
        _emit_replay_log(
            "fixed_scaffold_replay_failed",
            reason="optimizer_exception",
            error=f"{type(optimizer_exception).__name__}: {optimizer_exception}",
            objective_calls_total=int(objective_calls_total),
            elapsed_s=float(time.perf_counter() - t0),
        )
        return {
            "success": False,
            "available": False,
            "reason": "optimizer_exception",
            "error": f"{type(optimizer_exception).__name__}: {optimizer_exception}",
            "artifact_json": str(ctx["path"]),
            "structure_locked": True,
            "pool_type": str(pool_type),
        }
    if wallclock_hit and best_eval is None:
        _emit_replay_log(
            "fixed_scaffold_replay_failed",
            reason="wallclock_cap_before_first_eval",
            objective_calls_total=int(objective_calls_total),
            elapsed_s=float(time.perf_counter() - t0),
        )
        return {
            "success": False,
            "available": False,
            "reason": "wallclock_cap_before_first_eval",
            "artifact_json": str(ctx["path"]),
            "structure_locked": True,
            "pool_type": str(pool_type),
        }

    theta_best = (
        np.asarray(best_eval["theta_runtime"], dtype=float)
        if wallclock_hit and best_eval is not None
        else np.asarray(res.x if res is not None else theta0, dtype=float).reshape(-1)
    )
    stop_reason = "wallclock_cap" if wallclock_hit else "optimizer_complete"
    optimizer_success = bool(wallclock_hit or (res is not None and bool(res.success)))
    optimizer_message = "best-so-far returned at wallclock cap"
    optimizer_nfev = int(objective_calls_total)
    optimizer_nit = 0 if res is None else int(res.nit)
    if res is not None:
        optimizer_message = str(res.message)
        optimizer_nfev = int(res.nfev)

    initial_circuit = _build_ansatz_circuit(layout, theta0, int(nq), ref_state=ref_state)
    best_circuit = _build_ansatz_circuit(layout, theta_best, int(nq), ref_state=ref_state)
    initial_eval = _evaluate_locked_imported_circuit_energy(
        circuit=initial_circuit,
        ordered_labels_exyz=ordered_labels_exyz,
        static_coeff_map_exyz=static_coeff_map_exyz,
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        mitigation_config=mitigation_cfg,
        symmetry_mitigation_config=symmetry_cfg,
        backend_name=backend_name,
        use_fake_backend=bool(use_fake_backend),
        allow_aer_fallback=bool(allow_aer_fallback),
        omp_shm_workaround=bool(omp_shm_workaround),
        seed_transpiler=(None if seed_transpiler is None else int(seed_transpiler)),
        transpile_optimization_level=int(transpile_optimization_level),
        runtime_profile_config=runtime_profile_cfg,
        runtime_session_config=runtime_session_cfg,
    )
    best_runtime_energy_audits: dict[str, Any] | None = None
    best_eval_final = _evaluate_locked_imported_circuit_energy(
        circuit=best_circuit,
        ordered_labels_exyz=ordered_labels_exyz,
        static_coeff_map_exyz=static_coeff_map_exyz,
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        mitigation_config=mitigation_cfg,
        symmetry_mitigation_config=symmetry_cfg,
        backend_name=backend_name,
        use_fake_backend=bool(use_fake_backend),
        allow_aer_fallback=bool(allow_aer_fallback),
        omp_shm_workaround=bool(omp_shm_workaround),
        seed_transpiler=(None if seed_transpiler is None else int(seed_transpiler)),
        transpile_optimization_level=int(transpile_optimization_level),
        runtime_profile_config=runtime_profile_cfg,
        runtime_session_config=runtime_session_cfg,
    )
    theta0_logical = np.asarray(project_runtime_theta_block_mean(theta0, layout), dtype=float)
    theta_best_logical = np.asarray(project_runtime_theta_block_mean(theta_best, layout), dtype=float)
    return {
        "success": bool(optimizer_success),
        "available": True,
        "route": "fixed_lean_scaffold_noisy_replay",
        "artifact_json": str(ctx["path"]),
        "source_kind": str(ctx.get("source_kind", "unknown")),
        "pool_type": str(pool_type),
        "structure_locked": True,
        "matched_family_replay": False,
        "full_circuit_import_audit": False,
        "reps": 1,
        "reference_state_embedded": True,
        "ansatz_input_state_source": ansatz_input_state_meta.get("source", None),
        "ansatz_input_state_kind": ansatz_input_state_meta.get("handoff_state_kind", None),
        "parameterization": {
            "mode": "per_pauli_term_v1",
            "logical_parameter_count": int(layout.logical_parameter_count),
            "runtime_parameter_count": int(layout.runtime_parameter_count),
        },
        "optimizer": {
            "method": ("SPSA" if optimizer_method_key == "spsa" else "Powell"),
            "seed": int(optimizer_seed),
            "maxiter": int(optimizer_maxiter),
            "wallclock_cap_s": int(optimizer_wallclock_cap_s),
            "stop_reason": str(stop_reason),
            "iterations_completed": int(optimizer_nit),
            "objective_calls_total": int(optimizer_nfev),
            "message": str(optimizer_message),
        },
        "noise_config": {
            "noise_mode": str(noisy_mode),
            "shots": int(shots),
            "oracle_repeats": int(oracle_repeats),
            "oracle_aggregate": str(oracle_aggregate),
            "mitigation": dict(mitigation_cfg),
            "symmetry_mitigation": dict(symmetry_cfg),
            "runtime_profile": dict(runtime_profile_cfg),
            "runtime_session": dict(runtime_session_cfg),
        },
        "compile_control": {
            "transpile_optimization_level": int(transpile_optimization_level),
            "seed_transpiler": (None if seed_transpiler is None else int(seed_transpiler)),
            "source": ("runtime_profile_cli" if str(noisy_mode) == "runtime" else "legacy_route"),
        },
        "theta": {
            "initial_runtime": [float(x) for x in theta0.tolist()],
            "best_runtime": [float(x) for x in theta_best.tolist()],
            "initial_logical": [float(x) for x in theta0_logical.tolist()],
            "best_logical": [float(x) for x in theta_best_logical.tolist()],
        },
        "energies": {
            "saved_artifact_energy": ctx.get("saved_energy", None),
            "initial_noisy_mean": float(initial_eval["noisy_mean"]),
            "initial_noisy_stderr": float(initial_eval["noisy_stderr"]),
            "initial_ideal_mean": float(initial_eval["ideal_mean"]),
            "initial_ideal_stderr": float(initial_eval["ideal_stderr"]),
            "best_noisy_mean": float(best_eval_final["noisy_mean"]),
            "best_noisy_stderr": float(best_eval_final["noisy_stderr"]),
            "best_ideal_mean": float(best_eval_final["ideal_mean"]),
            "best_ideal_stderr": float(best_eval_final["ideal_stderr"]),
            "best_noisy_minus_ideal": float(best_eval_final["delta_mean"]),
            "best_noisy_minus_ideal_stderr": float(best_eval_final["delta_stderr"]),
            "best_ideal_minus_saved_artifact": (
                None
                if ctx.get("saved_energy", None) is None
                else float(best_eval_final["ideal_mean"]) - float(ctx["saved_energy"])
            ),
        },
        "objective_history_tail": list(objective_history_tail),
        "backend_info": dict(best_eval_final.get("backend_info", backend_info)),
        "elapsed_s": float(time.perf_counter() - t0),
    }


def _run_imported_fixed_lean_noise_attribution(
    *,
    artifact_json: str,
    shots: int,
    seed: int,
    oracle_repeats: int,
    oracle_aggregate: str,
    backend_name: str | None,
    use_fake_backend: bool,
    allow_aer_fallback: bool,
    omp_shm_workaround: bool,
    mitigation_config: dict[str, Any],
    symmetry_mitigation_config: dict[str, Any],
    slices: Sequence[str] = BACKEND_SCHEDULED_ATTRIBUTION_SLICES,
) -> dict[str, Any]:
    ctx, ansatz_input_state_meta, pool_type, error_payload = _resolve_locked_imported_lean_context(
        artifact_json,
        nonlean_reason="fixed_lean_attribution_requires_pareto_lean_l2_source",
    )
    if error_payload is not None or ctx is None:
        return dict(error_payload or {})

    mitigation_cfg = dict(normalize_mitigation_config(mitigation_config))
    symmetry_cfg = dict(normalize_symmetry_mitigation_config(symmetry_mitigation_config))
    if str(mitigation_cfg.get("mode", "none")) != "none":
        return {
            "success": False,
            "available": False,
            "reason": "fixed_lean_attribution_requires_mitigation_none",
            "artifact_json": str(ctx["path"]),
        }
    if str(symmetry_cfg.get("mode", "off")) != "off":
        return {
            "success": False,
            "available": False,
            "reason": "fixed_lean_attribution_requires_symmetry_off",
            "artifact_json": str(ctx["path"]),
        }

    requested_slices = tuple(str(x).strip().lower() for x in slices)
    qop = build_time_dependent_sparse_qop(
        ordered_labels_exyz=list(ctx["ordered_labels_exyz"]),
        static_coeff_map_exyz=dict(ctx["static_coeff_map_exyz"]),
        drive_coeff_map_exyz=None,
    )
    noisy_mode = "backend_scheduled"
    noisy_cfg = OracleConfig(
        noise_mode=str(noisy_mode),
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        backend_name=(None if backend_name is None else str(backend_name)),
        use_fake_backend=bool(use_fake_backend),
        allow_aer_fallback=bool(allow_aer_fallback),
        aer_fallback_mode="sampler_shots",
        omp_shm_workaround=bool(omp_shm_workaround),
        mitigation={"mode": "none", "zne_scales": [], "dd_sequence": None, "local_readout_strategy": None},
        symmetry_mitigation={"mode": "off"},
    )
    ideal_cfg = OracleConfig(
        noise_mode="ideal",
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        backend_name=None,
        use_fake_backend=False,
        mitigation={"mode": "none", "zne_scales": [], "dd_sequence": None, "local_readout_strategy": None},
        symmetry_mitigation={"mode": "off"},
    )

    t0 = time.perf_counter()
    with ExpectationOracle(noisy_cfg) as noisy_oracle, ExpectationOracle(ideal_cfg) as ideal_oracle:
        ideal_est = ideal_oracle.evaluate(ctx["circuit"], qop)
        attribution = noisy_oracle.evaluate_backend_scheduled_attribution(
            ctx["circuit"],
            qop,
            slices=requested_slices,
        )

    ideal_reference = {
        "mean": float(ideal_est.mean),
        "std": float(ideal_est.std),
        "stdev": float(ideal_est.stdev),
        "stderr": float(ideal_est.stderr),
        "n_samples": int(ideal_est.n_samples),
        "raw_values": [float(x) for x in ideal_est.raw_values],
        "aggregate": str(ideal_est.aggregate),
    }
    slice_payloads: dict[str, Any] = {}
    successful_slices: list[str] = []
    for slice_name in requested_slices:
        rec = attribution.get("slices", {}).get(str(slice_name), {})
        if not isinstance(rec, Mapping):
            slice_payloads[str(slice_name)] = {
                "success": False,
                "slice": str(slice_name),
                "components": {},
                "noisy_mean": None,
                "noisy_std": None,
                "noisy_stdev": None,
                "noisy_stderr": None,
                "ideal_mean": float(ideal_est.mean),
                "ideal_std": float(ideal_est.std),
                "ideal_stdev": float(ideal_est.stdev),
                "ideal_stderr": float(ideal_est.stderr),
                "delta_mean": None,
                "delta_stderr": None,
                "backend_info": None,
                "reason": "missing_slice_payload",
                "error": None,
            }
            continue
        est = rec.get("estimate", None)
        success = bool(rec.get("success", False) and est is not None)
        if success:
            successful_slices.append(str(slice_name))
        noisy_mean = None if est is None else float(est.mean)
        noisy_std = None if est is None else float(est.std)
        noisy_stdev = None if est is None else float(est.stdev)
        noisy_stderr = None if est is None else float(est.stderr)
        delta_mean = None if est is None else float(est.mean - ideal_est.mean)
        delta_stderr = None if est is None else float(_combine_stderr(est.stderr, ideal_est.stderr))
        slice_payloads[str(slice_name)] = {
            "success": bool(success),
            "slice": str(slice_name),
            "components": dict(rec.get("components", {})) if isinstance(rec.get("components", {}), Mapping) else {},
            "noisy_mean": noisy_mean,
            "noisy_std": noisy_std,
            "noisy_stdev": noisy_stdev,
            "noisy_stderr": noisy_stderr,
            "ideal_mean": float(ideal_est.mean),
            "ideal_std": float(ideal_est.std),
            "ideal_stdev": float(ideal_est.stdev),
            "ideal_stderr": float(ideal_est.stderr),
            "delta_mean": delta_mean,
            "delta_stderr": delta_stderr,
            "backend_info": dict(rec.get("backend_info", {})) if isinstance(rec.get("backend_info", {}), Mapping) else None,
            "reason": rec.get("reason", None),
            "error": rec.get("error", None),
        }

    full_delta = slice_payloads.get("full", {}).get("delta_mean", None)
    readout_delta = slice_payloads.get("readout_only", {}).get("delta_mean", None)
    gate_delta = slice_payloads.get("gate_stateprep_only", {}).get("delta_mean", None)
    full_noisy = slice_payloads.get("full", {}).get("noisy_mean", None)
    readout_noisy = slice_payloads.get("readout_only", {}).get("noisy_mean", None)
    gate_noisy = slice_payloads.get("gate_stateprep_only", {}).get("noisy_mean", None)

    def _maybe_diff(a: Any, b: Any) -> float | None:
        if a is None or b is None:
            return None
        return float(a) - float(b)

    return {
        "success": bool(len(successful_slices) == len(requested_slices)),
        "available": True,
        "route": "fixed_lean_noise_attribution",
        "artifact_json": str(ctx["path"]),
        "source_kind": str(ctx.get("source_kind", "unknown")),
        "pool_type": str(pool_type),
        "structure_locked": True,
        "matched_family_replay": False,
        "parameter_optimization": False,
        "reference_state_embedded": True,
        "ansatz_input_state_source": ansatz_input_state_meta.get("source", None),
        "ansatz_input_state_kind": ansatz_input_state_meta.get("handoff_state_kind", None),
        "parameterization": {
            "mode": "per_pauli_term_v1",
            "logical_parameter_count": int(ctx["layout"].logical_parameter_count),
            "runtime_parameter_count": int(ctx["layout"].runtime_parameter_count),
            "theta_source": "imported_artifact_runtime_theta",
        },
        "noise_config": {
            "noise_mode": str(noisy_mode),
            "shots": int(shots),
            "oracle_repeats": int(oracle_repeats),
            "oracle_aggregate": str(oracle_aggregate),
            "mitigation": dict(mitigation_cfg),
            "symmetry_mitigation": dict(symmetry_cfg),
        },
        "shared_compile": dict(attribution.get("shared_compile", {})),
        "ideal_reference": ideal_reference,
        "slices": slice_payloads,
        "slice_comparisons": {
            "full_minus_readout_only": _maybe_diff(full_noisy, readout_noisy),
            "full_minus_gate_stateprep_only": _maybe_diff(full_noisy, gate_noisy),
            "component_additivity_residual": (
                None
                if full_delta is None or readout_delta is None or gate_delta is None
                else float(full_delta) - float(readout_delta) - float(gate_delta)
            ),
        },
        "elapsed_s": float(time.perf_counter() - t0),
    }


def _locked_subject_payload_fields(subject: LockedImportedSubject) -> dict[str, Any]:
    return {
        "subject_kind": subject.subject_kind,
        "term_order_id": subject.term_order_id,
        "operator_count": subject.operator_count,
        "runtime_term_count": subject.runtime_term_count,
    }


def _extract_fixed_scaffold_compile_recommendation(ctx: Mapping[str, Any]) -> dict[str, Any] | None:
    payload = ctx.get("payload", {}) if isinstance(ctx, Mapping) else {}
    adapt_vqe = payload.get("adapt_vqe", {}) if isinstance(payload, Mapping) else {}
    fixed_meta = adapt_vqe.get("fixed_scaffold_metadata", {}) if isinstance(adapt_vqe, Mapping) else {}
    raw = fixed_meta.get("compile_recommendation", None) if isinstance(fixed_meta, Mapping) else None
    if not isinstance(raw, Mapping):
        return None
    return {
        "backend_name": None if raw.get("backend_name", None) is None else str(raw.get("backend_name")),
        "optimization_level": (
            None if raw.get("optimization_level", None) is None else int(raw.get("optimization_level"))
        ),
        "seed_transpiler": (
            None if raw.get("seed_transpiler", None) is None else int(raw.get("seed_transpiler"))
        ),
    }


def _run_imported_compile_control_scout_core(
    *,
    ctx: Mapping[str, Any],
    ansatz_input_state_meta: Mapping[str, Any],
    route: str,
    reason_prefix: str,
    pool_type: str | None,
    shots: int,
    seed: int,
    oracle_repeats: int,
    oracle_aggregate: str,
    backend_name: str | None,
    use_fake_backend: bool,
    allow_aer_fallback: bool,
    omp_shm_workaround: bool,
    mitigation_config: dict[str, Any],
    symmetry_mitigation_config: dict[str, Any],
    baseline_transpile_optimization_level: int,
    baseline_seed_transpiler: int,
    scout_transpile_optimization_levels: Sequence[int],
    scout_seed_transpilers: Sequence[int],
    rank_policy: str,
    extra_payload: Mapping[str, Any] | None = None,
    artifact_compile_recommendation: Mapping[str, Any] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    mitigation_cfg = dict(normalize_mitigation_config(mitigation_config))
    symmetry_cfg = dict(normalize_symmetry_mitigation_config(symmetry_mitigation_config))
    if str(mitigation_cfg.get("mode", "none")) != "readout":
        return {
            "success": False,
            "available": False,
            "reason": f"{reason_prefix}_requires_readout_mitigation",
            "artifact_json": str(ctx["path"]),
        }
    strategy = str(mitigation_cfg.get("local_readout_strategy") or "mthree")
    if strategy != "mthree":
        return {
            "success": False,
            "available": False,
            "reason": f"{reason_prefix}_requires_mthree",
            "artifact_json": str(ctx["path"]),
        }
    mitigation_cfg["local_readout_strategy"] = "mthree"
    if str(symmetry_cfg.get("mode", "off")) != "off":
        return {
            "success": False,
            "available": False,
            "reason": f"{reason_prefix}_requires_symmetry_off",
            "artifact_json": str(ctx["path"]),
        }
    if not bool(use_fake_backend):
        return {
            "success": False,
            "available": False,
            "reason": f"{reason_prefix}_requires_local_fake_backend",
            "artifact_json": str(ctx["path"]),
        }
    backend_name_norm = None if backend_name in {None, "", "none"} else str(backend_name)
    if backend_name_norm is None:
        return {
            "success": False,
            "available": False,
            "reason": f"{reason_prefix}_requires_backend_name",
            "artifact_json": str(ctx["path"]),
        }

    candidate_specs: list[dict[str, Any]] = []
    seen_specs: set[tuple[int, int]] = set()
    raw_specs = [
        (int(baseline_transpile_optimization_level), int(baseline_seed_transpiler), True),
        *[
            (int(opt_level), int(seed_trans), False)
            for opt_level in scout_transpile_optimization_levels
            for seed_trans in scout_seed_transpilers
        ],
    ]
    for opt_level, seed_trans, is_baseline in raw_specs:
        spec_key = (int(opt_level), int(seed_trans))
        if spec_key in seen_specs:
            continue
        seen_specs.add(spec_key)
        candidate_specs.append(
            {
                "transpile_optimization_level": int(opt_level),
                "seed_transpiler": int(seed_trans),
                "is_baseline": bool(is_baseline),
                "label": f"opt{int(opt_level)}_seed{int(seed_trans)}",
            }
        )

    def _emit_progress(event: str, **fields: Any) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(
                {
                    "event": str(event),
                    "route": str(route),
                    "artifact_json": str(ctx["path"]),
                    **fields,
                }
            )
        except Exception:
            pass

    def _candidate_counts_payload(candidates: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        successful = sum(1 for rec in candidates if bool(rec.get("success", False)))
        return {
            "total": int(len(candidate_specs)),
            "completed": int(len(candidates)),
            "successful": int(successful),
            "failed": int(len(candidates) - successful),
        }

    def _partial_payload(
        *,
        reason: str,
        candidates: Sequence[Mapping[str, Any]],
        last_candidate_label: str | None,
        last_candidate_index: int | None,
        elapsed_s: float,
    ) -> dict[str, Any]:
        candidate_records = [dict(rec) for rec in candidates]
        payload: dict[str, Any] = {
            "success": False,
            "available": True,
            "route": str(route),
            "reason": str(reason),
            "artifact_json": str(ctx["path"]),
            "pool_type": None if pool_type is None else str(pool_type),
            "candidate_counts": _candidate_counts_payload(candidate_records),
            "candidates_partial": candidate_records,
            "elapsed_s": float(elapsed_s),
            "last_candidate_label": (
                None if last_candidate_label in {None, ""} else str(last_candidate_label)
            ),
            "last_candidate_index": (
                None if last_candidate_index is None else int(last_candidate_index)
            ),
        }
        baseline_candidate = next(
            (dict(rec) for rec in candidate_records if bool(rec.get("is_baseline", False))),
            None,
        )
        if baseline_candidate is not None:
            payload["baseline_candidate"] = baseline_candidate
            payload["baseline_compile_observation"] = dict(
                baseline_candidate.get("compile_observation", {}) or {}
            )
        successful_candidates = [
            dict(rec) for rec in candidate_records if bool(rec.get("success", False))
        ]
        if successful_candidates:
            ranked_candidates = _rank_imported_compile_control_candidates(
                successful_candidates,
                rank_policy=str(rank_policy),
            )
            best_candidate = dict(ranked_candidates[0])
            payload["best_candidate"] = best_candidate
            payload["best_candidate_compile_observation"] = dict(
                best_candidate.get("compile_observation", {}) or {}
            )
        if artifact_compile_recommendation is not None:
            payload["artifact_compile_recommendation"] = dict(artifact_compile_recommendation)
        if isinstance(extra_payload, Mapping):
            payload.update(dict(extra_payload))
        return payload

    t0 = time.perf_counter()
    qop = _build_locked_imported_energy_qop(
        ordered_labels_exyz=list(ctx["ordered_labels_exyz"]),
        static_coeff_map_exyz=dict(ctx["static_coeff_map_exyz"]),
    )
    ideal_eval = _evaluate_locked_imported_circuit_ideal_energy(
        circuit=ctx["circuit"],
        ordered_labels_exyz=list(ctx["ordered_labels_exyz"]),
        static_coeff_map_exyz=dict(ctx["static_coeff_map_exyz"]),
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        qop=qop,
    )
    _ai_log(
        "compile_control_scout_ideal_ready",
        route=str(route),
        ideal_mean=float(ideal_eval["ideal_mean"]),
        candidate_total=int(len(candidate_specs)),
    )
    _emit_progress(
        "compile_control_scout_initialized",
        candidate_counts=_candidate_counts_payload(()),
        candidate_labels=[str(spec["label"]) for spec in candidate_specs],
        artifact_compile_recommendation=(
            None
            if artifact_compile_recommendation is None
            else dict(artifact_compile_recommendation)
        ),
        ideal_mean=float(ideal_eval["ideal_mean"]),
        ideal_stderr=float(ideal_eval["ideal_stderr"]),
        elapsed_s=float(time.perf_counter() - t0),
        partial_payload=_partial_payload(
            reason="in_progress",
            candidates=(),
            last_candidate_label=None,
            last_candidate_index=None,
            elapsed_s=float(time.perf_counter() - t0),
        ),
    )
    candidates: list[dict[str, Any]] = []
    for idx, spec in enumerate(candidate_specs):
        compile_request = _build_compile_request_payload(
            backend_name=backend_name_norm,
            seed_transpiler=int(spec["seed_transpiler"]),
            transpile_optimization_level=int(spec["transpile_optimization_level"]),
            source=f"{str(route)}_candidate",
        )
        _ai_log(
            "compile_control_scout_candidate_start",
            route=str(route),
            candidate_index=int(idx),
            candidate_label=str(spec["label"]),
            seed_transpiler=int(spec["seed_transpiler"]),
            transpile_optimization_level=int(spec["transpile_optimization_level"]),
            candidate_total=int(len(candidate_specs)),
        )
        _emit_progress(
            "compile_control_scout_candidate_started",
            candidate_index=int(idx),
            candidate_label=str(spec["label"]),
            compile_request=dict(compile_request),
            candidate_counts=_candidate_counts_payload(candidates),
            elapsed_s=float(time.perf_counter() - t0),
            partial_payload=_partial_payload(
                reason="in_progress",
                candidates=candidates,
                last_candidate_label=str(spec["label"]),
                last_candidate_index=int(idx),
                elapsed_s=float(time.perf_counter() - t0),
            ),
        )
        try:
            noisy_eval = _evaluate_locked_imported_circuit_noisy_energy(
                circuit=ctx["circuit"],
                ordered_labels_exyz=list(ctx["ordered_labels_exyz"]),
                static_coeff_map_exyz=dict(ctx["static_coeff_map_exyz"]),
                shots=int(shots),
                seed=int(seed),
                oracle_repeats=int(oracle_repeats),
                oracle_aggregate=str(oracle_aggregate),
                mitigation_config=mitigation_cfg,
                symmetry_mitigation_config=symmetry_cfg,
                backend_name=backend_name_norm,
                use_fake_backend=bool(use_fake_backend),
                allow_aer_fallback=bool(allow_aer_fallback),
                omp_shm_workaround=bool(omp_shm_workaround),
                seed_transpiler=int(spec["seed_transpiler"]),
                transpile_optimization_level=int(spec["transpile_optimization_level"]),
                qop=qop,
            )
            eval_payload = _combine_locked_imported_circuit_energy_evaluations(
                noisy_eval=noisy_eval,
                ideal_eval=ideal_eval,
            )
            backend_info = (
                dict(eval_payload.get("backend_info", {}))
                if isinstance(eval_payload.get("backend_info", {}), Mapping)
                else {}
            )
            compile_metrics = _extract_compile_metrics_from_backend_info(backend_info)
            compile_observation = _build_compile_observation_payload(
                requested=compile_request,
                backend_info=backend_info,
            )
            compile_metrics_flat = dict(compile_metrics)
            compile_metrics_flat.pop("transpile_seed", None)
            compile_metrics_flat.pop("transpile_optimization_level", None)
            candidates.append(
                {
                    "success": True,
                    "candidate_index": int(idx),
                    "label": str(spec["label"]),
                    "is_baseline": bool(spec["is_baseline"]),
                    "requested_backend_name": compile_request.get("backend_name", None),
                    "requested_seed_transpiler": compile_request.get("seed_transpiler", None),
                    "requested_transpile_optimization_level": compile_request.get(
                        "transpile_optimization_level", None
                    ),
                    "compile_request": dict(compile_request),
                    "compile_observation": dict(compile_observation),
                    "transpile_seed": compile_metrics.get("transpile_seed", None),
                    "transpile_optimization_level": compile_metrics.get(
                        "transpile_optimization_level", None
                    ),
                    "noisy_mean": float(eval_payload["noisy_mean"]),
                    "noisy_stderr": float(eval_payload["noisy_stderr"]),
                    "ideal_mean": float(eval_payload["ideal_mean"]),
                    "ideal_stderr": float(eval_payload["ideal_stderr"]),
                    "delta_mean": float(eval_payload["delta_mean"]),
                    "delta_stderr": float(eval_payload["delta_stderr"]),
                    "delta_abs": abs(float(eval_payload["delta_mean"])),
                    "backend_info": backend_info,
                    **compile_metrics_flat,
                }
            )
        except Exception as exc:
            candidates.append(
                {
                    "success": False,
                    "candidate_index": int(idx),
                    "label": str(spec["label"]),
                    "is_baseline": bool(spec["is_baseline"]),
                    "requested_backend_name": compile_request.get("backend_name", None),
                    "requested_seed_transpiler": compile_request.get("seed_transpiler", None),
                    "requested_transpile_optimization_level": compile_request.get(
                        "transpile_optimization_level", None
                    ),
                    "compile_request": dict(compile_request),
                    "compile_observation": {
                        "available": False,
                        "requested": dict(compile_request),
                        "observed": None,
                        "matches_requested": None,
                        "mismatch_fields": [],
                        "reason": "candidate_exception",
                    },
                    "transpile_seed": int(spec["seed_transpiler"]),
                    "transpile_optimization_level": int(spec["transpile_optimization_level"]),
                    "reason": "candidate_exception",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        last_candidate = dict(candidates[-1])
        _ai_log(
            "compile_control_scout_candidate_done",
            route=str(route),
            candidate_index=int(idx),
            candidate_label=str(spec["label"]),
            success=bool(last_candidate.get("success", False)),
            delta_mean=last_candidate.get("delta_mean", None),
            compiled_two_qubit_count=last_candidate.get("compiled_two_qubit_count", None),
            compiled_depth=last_candidate.get("compiled_depth", None),
            elapsed_s=float(time.perf_counter() - t0),
        )
        _emit_progress(
            "compile_control_scout_candidate_completed",
            candidate_index=int(idx),
            candidate_label=str(spec["label"]),
            candidate=last_candidate,
            candidate_counts=_candidate_counts_payload(candidates),
            elapsed_s=float(time.perf_counter() - t0),
            partial_payload=_partial_payload(
                reason="in_progress",
                candidates=candidates,
                last_candidate_label=str(spec["label"]),
                last_candidate_index=int(idx),
                elapsed_s=float(time.perf_counter() - t0),
            ),
        )

    successful_candidates = [dict(rec) for rec in candidates if bool(rec.get("success", False))]
    if not successful_candidates:
        return _partial_payload(
            reason=f"{reason_prefix}_no_successful_candidates",
            candidates=candidates,
            last_candidate_label=(
                None if not candidates else str(candidates[-1].get("label", ""))
            ),
            last_candidate_index=(
                None if not candidates else int(candidates[-1].get("candidate_index", 0))
            ),
            elapsed_s=float(time.perf_counter() - t0),
        )

    ranked_candidates = _rank_imported_compile_control_candidates(
        successful_candidates,
        rank_policy=str(rank_policy),
    )
    best_candidate = dict(ranked_candidates[0])
    baseline_candidate = next(
        (dict(rec) for rec in candidates if bool(rec.get("is_baseline", False))),
        None,
    )
    baseline_success = bool(isinstance(baseline_candidate, Mapping) and baseline_candidate.get("success", False))

    baseline_delta_abs = (
        None
        if not baseline_success or baseline_candidate is None
        else abs(float(baseline_candidate.get("delta_mean", float("nan"))))
    )
    best_delta_abs = abs(float(best_candidate.get("delta_mean", float("nan"))))

    def _maybe_delta_int(a: Any, b: Any) -> int | None:
        if a is None or b is None:
            return None
        return int(a) - int(b)

    ranking_summary = {
        "policy": str(rank_policy),
        "candidate_labels_ranked": [str(rec.get("label", "")) for rec in ranked_candidates],
        "best_label": str(best_candidate.get("label", "")),
        "baseline_label": (
            None if baseline_candidate is None else str(baseline_candidate.get("label", ""))
        ),
        "best_vs_baseline_delta_abs_improvement": (
            None
            if baseline_delta_abs is None
            else float(baseline_delta_abs) - float(best_delta_abs)
        ),
        "best_vs_baseline_two_qubit_delta": _maybe_delta_int(
            best_candidate.get("compiled_two_qubit_count", None),
            None if baseline_candidate is None else baseline_candidate.get("compiled_two_qubit_count", None),
        ),
        "best_vs_baseline_depth_delta": _maybe_delta_int(
            best_candidate.get("compiled_depth", None),
            None if baseline_candidate is None else baseline_candidate.get("compiled_depth", None),
        ),
        "best_vs_baseline_size_delta": _maybe_delta_int(
            best_candidate.get("compiled_size", None),
            None if baseline_candidate is None else baseline_candidate.get("compiled_size", None),
        ),
    }

    payload = {
        "success": bool(baseline_success),
        "available": True,
        "route": str(route),
        "artifact_json": str(ctx["path"]),
        "source_kind": str(ctx.get("source_kind", "unknown")),
        "pool_type": None if pool_type is None else str(pool_type),
        "structure_locked": True,
        "matched_family_replay": False,
        "parameter_optimization": False,
        "reference_state_embedded": True,
        "ansatz_input_state_source": ansatz_input_state_meta.get("source", None),
        "ansatz_input_state_kind": ansatz_input_state_meta.get("handoff_state_kind", None),
        "parameterization": {
            "mode": "per_pauli_term_v1",
            "logical_parameter_count": int(ctx["layout"].logical_parameter_count),
            "runtime_parameter_count": int(ctx["layout"].runtime_parameter_count),
            "theta_source": "imported_artifact_runtime_theta",
        },
        "noise_config": {
            "noise_mode": "backend_scheduled",
            "shots": int(shots),
            "oracle_repeats": int(oracle_repeats),
            "oracle_aggregate": str(oracle_aggregate),
            "backend_name": str(backend_name_norm),
            "use_fake_backend": True,
            "mitigation": dict(mitigation_cfg),
            "symmetry_mitigation": dict(symmetry_cfg),
        },
        "compile_control_config": {
            "baseline_transpile_optimization_level": int(baseline_transpile_optimization_level),
            "baseline_seed_transpiler": int(baseline_seed_transpiler),
            "scout_transpile_optimization_levels": [
                int(x) for x in scout_transpile_optimization_levels
            ],
            "scout_seed_transpilers": [int(x) for x in scout_seed_transpilers],
            "rank_policy": str(rank_policy),
        },
        "candidate_counts": {
            "total": int(len(candidates)),
            "successful": int(len(successful_candidates)),
            "failed": int(len(candidates) - len(successful_candidates)),
        },
        "baseline_candidate": baseline_candidate,
        "baseline_compile_request": (
            None
            if not isinstance(baseline_candidate, Mapping)
            else dict(baseline_candidate.get("compile_request", {}) or {})
        ),
        "baseline_compile_observation": (
            None
            if not isinstance(baseline_candidate, Mapping)
            else dict(baseline_candidate.get("compile_observation", {}) or {})
        ),
        "best_candidate": best_candidate,
        "best_candidate_compile_observation": dict(
            best_candidate.get("compile_observation", {}) or {}
        ),
        "candidates": candidates,
        "ranking": ranking_summary,
        "elapsed_s": float(time.perf_counter() - t0),
    }
    if artifact_compile_recommendation is not None:
        payload["artifact_compile_recommendation"] = dict(artifact_compile_recommendation)
    if isinstance(extra_payload, Mapping):
        payload.update(dict(extra_payload))
    return payload


def _run_imported_fixed_lean_compile_control_scout(
    *,
    artifact_json: str,
    shots: int,
    seed: int,
    oracle_repeats: int,
    oracle_aggregate: str,
    backend_name: str | None,
    use_fake_backend: bool,
    allow_aer_fallback: bool,
    omp_shm_workaround: bool,
    mitigation_config: dict[str, Any],
    symmetry_mitigation_config: dict[str, Any],
    baseline_transpile_optimization_level: int,
    baseline_seed_transpiler: int,
    scout_transpile_optimization_levels: Sequence[int],
    scout_seed_transpilers: Sequence[int],
    rank_policy: str,
) -> dict[str, Any]:
    ctx, ansatz_input_state_meta, pool_type, error_payload = _resolve_locked_imported_lean_context(
        artifact_json,
        nonlean_reason="fixed_lean_compile_control_scout_requires_pareto_lean_l2_source",
    )
    if error_payload is not None or ctx is None:
        return dict(error_payload or {})
    return _run_imported_compile_control_scout_core(
        ctx=ctx,
        ansatz_input_state_meta=ansatz_input_state_meta,
        route="fixed_lean_compile_control_scout",
        reason_prefix="fixed_lean_compile_control_scout",
        pool_type=(None if pool_type is None else str(pool_type)),
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        backend_name=backend_name,
        use_fake_backend=bool(use_fake_backend),
        allow_aer_fallback=bool(allow_aer_fallback),
        omp_shm_workaround=bool(omp_shm_workaround),
        mitigation_config=dict(mitigation_config),
        symmetry_mitigation_config=dict(symmetry_mitigation_config),
        baseline_transpile_optimization_level=int(baseline_transpile_optimization_level),
        baseline_seed_transpiler=int(baseline_seed_transpiler),
        scout_transpile_optimization_levels=list(scout_transpile_optimization_levels),
        scout_seed_transpilers=list(scout_seed_transpilers),
        rank_policy=str(rank_policy),
    )


def _run_imported_fixed_scaffold_compile_control_scout(
    *,
    artifact_json: str,
    shots: int,
    seed: int,
    oracle_repeats: int,
    oracle_aggregate: str,
    backend_name: str | None,
    use_fake_backend: bool,
    allow_aer_fallback: bool,
    omp_shm_workaround: bool,
    mitigation_config: dict[str, Any],
    symmetry_mitigation_config: dict[str, Any],
    baseline_transpile_optimization_level: int,
    baseline_seed_transpiler: int,
    scout_transpile_optimization_levels: Sequence[int],
    scout_seed_transpilers: Sequence[int],
    rank_policy: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    ctx, ansatz_input_state_meta, subject, error_payload = _resolve_locked_imported_fixed_scaffold_context(
        artifact_json,
        nonfixed_reason="fixed_scaffold_compile_control_scout_requires_locked_fixed_scaffold_source",
    )
    if error_payload is not None or ctx is None or subject is None:
        return dict(error_payload or {})
    return _run_imported_compile_control_scout_core(
        ctx=ctx,
        ansatz_input_state_meta=ansatz_input_state_meta,
        route="fixed_scaffold_compile_control_scout",
        reason_prefix="fixed_scaffold_compile_control_scout",
        pool_type=(None if subject.pool_type is None else str(subject.pool_type)),
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        backend_name=backend_name,
        use_fake_backend=bool(use_fake_backend),
        allow_aer_fallback=bool(allow_aer_fallback),
        omp_shm_workaround=bool(omp_shm_workaround),
        mitigation_config=dict(mitigation_config),
        symmetry_mitigation_config=dict(symmetry_mitigation_config),
        baseline_transpile_optimization_level=int(baseline_transpile_optimization_level),
        baseline_seed_transpiler=int(baseline_seed_transpiler),
        scout_transpile_optimization_levels=list(scout_transpile_optimization_levels),
        scout_seed_transpilers=list(scout_seed_transpilers),
        rank_policy=str(rank_policy),
        extra_payload=_locked_subject_payload_fields(subject),
        artifact_compile_recommendation=_extract_fixed_scaffold_compile_recommendation(ctx),
        progress_callback=progress_callback,
    )


def _run_imported_fixed_scaffold_runtime_energy_only(
    *,
    artifact_json: str,
    shots: int,
    seed: int,
    oracle_repeats: int,
    oracle_aggregate: str,
    backend_name: str | None,
    use_fake_backend: bool,
    allow_aer_fallback: bool,
    omp_shm_workaround: bool,
    mitigation_config: dict[str, Any],
    symmetry_mitigation_config: dict[str, Any],
    runtime_profile_config: dict[str, Any],
    runtime_session_config: dict[str, Any],
    transpile_optimization_level: int,
    seed_transpiler: int | None,
    include_dd_probe: bool,
    include_final_zne_audit: bool,
) -> dict[str, Any]:
    ctx, ansatz_input_state_meta, subject, error_payload = _resolve_locked_imported_fixed_scaffold_context(
        artifact_json,
        nonfixed_reason="fixed_scaffold_runtime_energy_only_requires_locked_fixed_scaffold_source",
    )
    if error_payload is not None or ctx is None or subject is None:
        return dict(error_payload or {})
    if bool(use_fake_backend):
        return {
            "success": False,
            "available": False,
            "reason": "fixed_scaffold_runtime_energy_only_requires_runtime_backend",
            "artifact_json": str(ctx["path"]),
            **_locked_subject_payload_fields(subject),
        }
    if backend_name in {None, ""}:
        return {
            "success": False,
            "available": False,
            "reason": "fixed_scaffold_runtime_energy_only_requires_backend_name",
            "artifact_json": str(ctx["path"]),
            **_locked_subject_payload_fields(subject),
        }

    mitigation_cfg = dict(normalize_mitigation_config(mitigation_config))
    symmetry_cfg = dict(normalize_symmetry_mitigation_config(symmetry_mitigation_config))
    if str(mitigation_cfg.get("mode", "none")) != "readout":
        return {
            "success": False,
            "available": False,
            "reason": "fixed_scaffold_runtime_energy_only_requires_readout_mitigation",
            "artifact_json": str(ctx["path"]),
            **_locked_subject_payload_fields(subject),
        }
    mitigation_cfg["local_readout_strategy"] = "mthree"
    if str(symmetry_cfg.get("mode", "off")) != "off":
        return {
            "success": False,
            "available": False,
            "reason": "fixed_scaffold_runtime_energy_only_requires_symmetry_off",
            "artifact_json": str(ctx["path"]),
            **_locked_subject_payload_fields(subject),
        }

    runtime_profile_cfg = dict(normalize_runtime_estimator_profile_config(runtime_profile_config))
    runtime_session_cfg = dict(normalize_runtime_session_policy_config(runtime_session_config))
    layout = ctx["layout"]
    nq = int(ctx["num_qubits"])
    theta_runtime = np.asarray(ctx["theta_runtime"], dtype=float).reshape(-1)
    ref_state = np.asarray(ctx["ansatz_input_state"], dtype=complex).reshape(-1)
    ordered_labels_exyz = list(ctx["ordered_labels_exyz"])
    static_coeff_map_exyz = dict(ctx["static_coeff_map_exyz"])
    circuit = _build_ansatz_circuit(layout, theta_runtime, int(nq), ref_state=ref_state)
    phase_evals = _run_locked_imported_runtime_phase_evals(
        circuit=circuit,
        ordered_labels_exyz=ordered_labels_exyz,
        static_coeff_map_exyz=static_coeff_map_exyz,
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        mitigation_config=mitigation_cfg,
        symmetry_mitigation_config=symmetry_cfg,
        backend_name=str(backend_name),
        allow_aer_fallback=bool(allow_aer_fallback),
        omp_shm_workaround=bool(omp_shm_workaround),
        seed_transpiler=(None if seed_transpiler is None else int(seed_transpiler)),
        transpile_optimization_level=int(transpile_optimization_level),
        runtime_session_config=runtime_session_cfg,
        main_runtime_profile_config=runtime_profile_cfg,
        dd_probe_runtime_profile_config=(
            normalize_runtime_estimator_profile_config("dd_probe_twirled_readout_v1")
            if bool(include_dd_probe)
            else None
        ),
        final_audit_runtime_profile_config=(
            normalize_runtime_estimator_profile_config("final_audit_zne_twirled_readout_v1")
            if bool(include_final_zne_audit)
            else None
        ),
    )
    main_eval = phase_evals.get("main", {}) if isinstance(phase_evals, Mapping) else {}
    main_eval_payload = main_eval.get("evaluation", {}) if isinstance(main_eval, Mapping) else {}
    return {
        "success": bool(isinstance(main_eval, Mapping) and main_eval.get("success", False)),
        "available": True,
        "route": "fixed_scaffold_runtime_energy_only",
        "artifact_json": str(ctx["path"]),
        "candidate_artifact_json": str(ctx["path"]),
        "source_kind": str(ctx.get("source_kind", "unknown")),
        "pool_type": str(subject.pool_type),
        "structure_locked": True,
        "reference_state_embedded": True,
        "ansatz_input_state_source": ansatz_input_state_meta.get("source", None),
        "ansatz_input_state_kind": ansatz_input_state_meta.get("handoff_state_kind", None),
        "noise_config": {
            "noise_mode": "runtime",
            "shots": int(shots),
            "oracle_repeats": int(oracle_repeats),
            "oracle_aggregate": str(oracle_aggregate),
            "backend_name": str(backend_name),
            "mitigation": dict(mitigation_cfg),
            "symmetry_mitigation": dict(symmetry_cfg),
            "runtime_profile": dict(runtime_profile_cfg),
            "runtime_session": dict(runtime_session_cfg),
        },
        "compile_control": {
            "transpile_optimization_level": int(transpile_optimization_level),
            "seed_transpiler": (None if seed_transpiler is None else int(seed_transpiler)),
            "source": "runtime_profile_cli",
        },
        "energy_audits": dict(phase_evals),
        "backend_info": dict(main_eval_payload.get("backend_info", {})) if isinstance(main_eval_payload, Mapping) else {},
        **_locked_subject_payload_fields(subject),
    }


def _run_imported_fixed_scaffold_runtime_raw_baseline(
    *,
    artifact_json: str,
    shots: int,
    seed: int,
    oracle_repeats: int,
    oracle_aggregate: str,
    backend_name: str | None,
    use_fake_backend: bool,
    allow_aer_fallback: bool,
    omp_shm_workaround: bool,
    mitigation_config: dict[str, Any],
    symmetry_mitigation_config: dict[str, Any],
    runtime_profile_config: dict[str, Any],
    runtime_session_config: dict[str, Any],
    transpile_optimization_level: int,
    seed_transpiler: int | None,
    raw_transport: str = "auto",
    raw_store_memory: bool = False,
    raw_artifact_path: str | None = None,
) -> dict[str, Any]:
    ctx, ansatz_input_state_meta, subject, error_payload = _resolve_locked_imported_fixed_scaffold_context(
        artifact_json,
        nonfixed_reason="fixed_scaffold_runtime_raw_baseline_requires_locked_fixed_scaffold_source",
    )
    if error_payload is not None or ctx is None or subject is None:
        return dict(error_payload or {})
    requested_mitigation_cfg = dict(normalize_mitigation_config(mitigation_config))
    requested_symmetry_cfg = dict(normalize_symmetry_mitigation_config(symmetry_mitigation_config))
    runtime_profile_cfg = dict(normalize_runtime_estimator_profile_config(runtime_profile_config))
    runtime_session_cfg = dict(normalize_runtime_session_policy_config(runtime_session_config))

    symmetry_meta = _resolve_locked_imported_hh_symmetry_metadata(ctx)

    effective_requested_symmetry_cfg = dict(requested_symmetry_cfg)
    if symmetry_meta is not None:
        effective_requested_symmetry_cfg.update(
            {
                "num_sites": int(symmetry_meta["num_sites"]),
                "ordering": str(symmetry_meta["ordering"]),
                "sector_n_up": int(symmetry_meta["sector_n_up"]),
                "sector_n_dn": int(symmetry_meta["sector_n_dn"]),
            }
        )
    effective_num_sites = effective_requested_symmetry_cfg.get("num_sites", None)
    effective_ordering = effective_requested_symmetry_cfg.get("ordering", None)
    effective_sector_n_up = effective_requested_symmetry_cfg.get("sector_n_up", None)
    effective_sector_n_dn = effective_requested_symmetry_cfg.get("sector_n_dn", None)

    raw_transport_key = str(raw_transport).strip().lower() or "auto"
    if bool(use_fake_backend):
        backend_name = "FakeMarrakesh" if backend_name in {None, ""} else str(backend_name)
        requested_mitigation_mode = str(requested_mitigation_cfg.get("mode", "none"))
        if requested_mitigation_mode not in {"none", "readout"}:
            return {
                "success": False,
                "available": False,
                "reason": "fixed_scaffold_runtime_raw_baseline_local_postprocessing_requires_none_or_readout",
                "artifact_json": str(ctx["path"]),
                **_locked_subject_payload_fields(subject),
            }
        if requested_mitigation_mode == "readout" and requested_mitigation_cfg.get(
            "local_readout_strategy", None
        ) in {None, "", "none"}:
            requested_mitigation_cfg["local_readout_strategy"] = "mthree"
        if requested_mitigation_mode == "readout" and str(
            requested_mitigation_cfg.get("local_readout_strategy", "mthree")
        ) != "mthree":
            return {
                "success": False,
                "available": False,
                "reason": "fixed_scaffold_runtime_raw_baseline_local_postprocessing_requires_mthree",
                "artifact_json": str(ctx["path"]),
                **_locked_subject_payload_fields(subject),
            }
        if requested_mitigation_cfg.get("zne_scales", []):
            return {
                "success": False,
                "available": False,
                "reason": "fixed_scaffold_runtime_raw_baseline_local_postprocessing_rejects_zne",
                "artifact_json": str(ctx["path"]),
                **_locked_subject_payload_fields(subject),
            }
        if requested_mitigation_cfg.get("dd_sequence", None) not in {None, "", "none"}:
            return {
                "success": False,
                "available": False,
                "reason": "fixed_scaffold_runtime_raw_baseline_local_postprocessing_rejects_dd",
                "artifact_json": str(ctx["path"]),
                **_locked_subject_payload_fields(subject),
            }
        if bool(requested_mitigation_cfg.get("local_gate_twirling", False)):
            return {
                "success": False,
                "available": False,
                "reason": "fixed_scaffold_runtime_raw_baseline_local_postprocessing_rejects_gate_twirling",
                "artifact_json": str(ctx["path"]),
                **_locked_subject_payload_fields(subject),
            }
        if str(effective_requested_symmetry_cfg.get("mode", "off")) not in {
            "off",
            "verify_only",
            "postselect_diag_v1",
            "projector_renorm_v1",
        }:
            return {
                "success": False,
                "available": False,
                "reason": "fixed_scaffold_runtime_raw_baseline_local_postprocessing_requires_diagonal_symmetry",
                "artifact_json": str(ctx["path"]),
                **_locked_subject_payload_fields(subject),
            }
        if raw_transport_key == "sampler_v2":
            return {
                "success": False,
                "available": False,
                "reason": "fixed_scaffold_runtime_raw_baseline_requires_backend_run_transport",
                "artifact_json": str(ctx["path"]),
                **_locked_subject_payload_fields(subject),
            }
        acquisition_mitigation_cfg = dict(normalize_mitigation_config({"mode": "none"}))
        acquisition_symmetry_cfg = dict(
            normalize_symmetry_mitigation_config({"mode": "off"})
        )
        acquisition_noise_mode = "backend_scheduled"
    else:
        if backend_name in {None, ""}:
            return {
                "success": False,
                "available": False,
                "reason": "fixed_scaffold_runtime_raw_baseline_requires_backend_name",
                "artifact_json": str(ctx["path"]),
                **_locked_subject_payload_fields(subject),
            }
        if str(requested_mitigation_cfg.get("mode", "none")) != "none":
            return {
                "success": False,
                "available": False,
                "reason": "fixed_scaffold_runtime_raw_baseline_requires_no_mitigation",
                "artifact_json": str(ctx["path"]),
                **_locked_subject_payload_fields(subject),
            }
        if str(effective_requested_symmetry_cfg.get("mode", "off")) not in {
            "off",
            "verify_only",
        }:
            return {
                "success": False,
                "available": False,
                "reason": "fixed_scaffold_runtime_raw_baseline_requires_off_or_verify_only_symmetry",
                "artifact_json": str(ctx["path"]),
                **_locked_subject_payload_fields(subject),
            }
        try:
            normalize_sampler_raw_runtime_config(
                OracleConfig(
                    noise_mode="runtime",
                    shots=int(shots),
                    seed=int(seed),
                    seed_transpiler=(
                        None if seed_transpiler is None else int(seed_transpiler)
                    ),
                    transpile_optimization_level=int(transpile_optimization_level),
                    oracle_repeats=int(oracle_repeats),
                    oracle_aggregate=str(oracle_aggregate),
                    backend_name=str(backend_name),
                    use_fake_backend=False,
                    allow_aer_fallback=bool(allow_aer_fallback),
                    aer_fallback_mode="sampler_shots",
                    omp_shm_workaround=bool(omp_shm_workaround),
                    mitigation=dict(requested_mitigation_cfg),
                    symmetry_mitigation=dict(effective_requested_symmetry_cfg),
                    runtime_profile=dict(runtime_profile_cfg),
                    runtime_session=dict(runtime_session_cfg),
                    execution_surface="raw_measurement_v1",
                    raw_transport=str(raw_transport),
                    raw_store_memory=bool(raw_store_memory),
                    raw_artifact_path=(
                        None if raw_artifact_path in {None, ""} else str(raw_artifact_path)
                    ),
                )
            )
        except ValueError as exc:
            msg = str(exc)
            if "raw_transport" in msg:
                reason = "fixed_scaffold_runtime_raw_baseline_requires_sampler_transport"
            elif "backend_name" in msg:
                reason = "fixed_scaffold_runtime_raw_baseline_requires_backend_name"
            elif "use_fake_backend=False" in msg or "noise_mode='runtime'" in msg:
                reason = "fixed_scaffold_runtime_raw_baseline_requires_runtime_backend"
            elif "mitigation_mode='none'" in msg:
                reason = "fixed_scaffold_runtime_raw_baseline_requires_no_mitigation"
            elif "symmetry_mitigation in {'off','verify_only'}" in msg:
                reason = "fixed_scaffold_runtime_raw_baseline_requires_off_or_verify_only_symmetry"
            elif "runtime_profile in" in msg or "suppression-only runtime profiles" in msg:
                reason = "fixed_scaffold_runtime_raw_baseline_requires_sampler_safe_runtime_profile"
            else:
                reason = "fixed_scaffold_runtime_raw_baseline_invalid_sampler_runtime_config"
            return {
                "success": False,
                "available": False,
                "reason": str(reason),
                "artifact_json": str(ctx["path"]),
                **_locked_subject_payload_fields(subject),
            }
        acquisition_mitigation_cfg = dict(requested_mitigation_cfg)
        acquisition_symmetry_cfg = dict(effective_requested_symmetry_cfg)
        acquisition_noise_mode = "runtime"

    layout = ctx["layout"]
    nq = int(ctx["num_qubits"])
    theta_runtime = np.asarray(ctx["theta_runtime"], dtype=float).reshape(-1)
    ref_state = np.asarray(ctx["ansatz_input_state"], dtype=complex).reshape(-1)
    ordered_labels_exyz = list(ctx["ordered_labels_exyz"])
    static_coeff_map_exyz = dict(ctx["static_coeff_map_exyz"])
    plan = build_parameterized_ansatz_plan(layout, nq=int(nq), ref_state=ref_state)

    main_eval = _evaluate_locked_imported_circuit_raw_energy(
        plan=plan,
        theta_runtime=theta_runtime,
        ordered_labels_exyz=ordered_labels_exyz,
        static_coeff_map_exyz=static_coeff_map_exyz,
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        mitigation_config=acquisition_mitigation_cfg,
        symmetry_mitigation_config=acquisition_symmetry_cfg,
        backend_name=(None if backend_name in {None, ""} else str(backend_name)),
        use_fake_backend=bool(use_fake_backend),
        allow_aer_fallback=bool(allow_aer_fallback),
        omp_shm_workaround=bool(omp_shm_workaround),
        seed_transpiler=(None if seed_transpiler is None else int(seed_transpiler)),
        transpile_optimization_level=int(transpile_optimization_level),
        runtime_profile_config=runtime_profile_cfg,
        runtime_session_config=runtime_session_cfg,
        raw_transport=str(raw_transport),
        raw_store_memory=bool(raw_store_memory),
        raw_artifact_path=(None if raw_artifact_path in {None, ""} else str(raw_artifact_path)),
    )
    main_compile_signature = _runtime_compile_signature(main_eval.get("backend_info", {}))
    symmetry_diagnostic = _evaluate_locked_imported_circuit_raw_symmetry_diagnostic(
        plan=plan,
        theta_runtime=theta_runtime,
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        mitigation_config=acquisition_mitigation_cfg,
        symmetry_mitigation_config=acquisition_symmetry_cfg,
        backend_name=(None if backend_name in {None, ""} else str(backend_name)),
        use_fake_backend=bool(use_fake_backend),
        allow_aer_fallback=bool(allow_aer_fallback),
        omp_shm_workaround=bool(omp_shm_workaround),
        num_sites=(None if effective_num_sites is None else int(effective_num_sites)),
        ordering=(None if effective_ordering is None else str(effective_ordering)),
        sector_n_up=(None if effective_sector_n_up is None else int(effective_sector_n_up)),
        sector_n_dn=(None if effective_sector_n_dn is None else int(effective_sector_n_dn)),
        seed_transpiler=(None if seed_transpiler is None else int(seed_transpiler)),
        transpile_optimization_level=int(transpile_optimization_level),
        runtime_profile_config=runtime_profile_cfg,
        runtime_session_config=runtime_session_cfg,
        raw_transport=str(raw_transport),
        raw_store_memory=bool(raw_store_memory),
        raw_artifact_path=(None if raw_artifact_path in {None, ""} else str(raw_artifact_path)),
        diagonal_postprocessing_mitigation_config=dict(requested_mitigation_cfg),
        diagonal_postprocessing_symmetry_config=dict(effective_requested_symmetry_cfg),
    )
    symmetry_validation = _evaluate_locked_imported_circuit_raw_symmetry_validation(
        plan=plan,
        theta_runtime=theta_runtime,
        symmetry_diagnostic=symmetry_diagnostic,
        num_sites=(None if effective_num_sites is None else int(effective_num_sites)),
        ordering=(None if effective_ordering is None else str(effective_ordering)),
        sector_n_up=(None if effective_sector_n_up is None else int(effective_sector_n_up)),
        sector_n_dn=(None if effective_sector_n_dn is None else int(effective_sector_n_dn)),
    )
    symmetry_bootstrap = _evaluate_locked_imported_circuit_raw_symmetry_bootstrap(
        symmetry_diagnostic=symmetry_diagnostic,
        num_sites=(None if effective_num_sites is None else int(effective_num_sites)),
        ordering=(None if effective_ordering is None else str(effective_ordering)),
        sector_n_up=(None if effective_sector_n_up is None else int(effective_sector_n_up)),
        sector_n_dn=(None if effective_sector_n_dn is None else int(effective_sector_n_dn)),
        oracle_repeats=int(oracle_repeats),
        seed=int(seed),
    )
    diagonal_postprocessing = (
        symmetry_diagnostic.get("diagonal_postprocessing", {})
        if isinstance(symmetry_diagnostic, Mapping)
        and isinstance(symmetry_diagnostic.get("diagonal_postprocessing", {}), Mapping)
        else {
            "success": False,
            "available": False,
            "reason": "diagonal_postprocessing_unavailable",
            "summary": None,
            "readout_details": None,
            "error_type": None,
            "error_message": None,
        }
    )
    diagonal_postprocessing = {
        **dict(diagonal_postprocessing),
        "validation": _evaluate_locked_imported_circuit_raw_diagonal_postprocessing_validation(
            plan=plan,
            theta_runtime=theta_runtime,
            diagonal_postprocessing=diagonal_postprocessing,
            num_sites=(None if effective_num_sites is None else int(effective_num_sites)),
            ordering=(None if effective_ordering is None else str(effective_ordering)),
            sector_n_up=(None if effective_sector_n_up is None else int(effective_sector_n_up)),
            sector_n_dn=(None if effective_sector_n_dn is None else int(effective_sector_n_dn)),
        ),
    }
    phase_evals = {
        "main": {
            "success": True,
            "profile": dict(runtime_profile_cfg),
            "evaluation": dict(main_eval),
            "compile_signature": dict(main_compile_signature),
            "symmetry_diagnostic": dict(symmetry_diagnostic),
            "symmetry_validation": dict(symmetry_validation),
            "symmetry_bootstrap": dict(symmetry_bootstrap),
            "diagonal_postprocessing": dict(diagonal_postprocessing),
        },
        "dd_probe": {
            "enabled": False,
            "success": False,
            "reason": "raw_acquisition_only_v1",
        },
        "final_audit_zne": {
            "enabled": False,
            "success": False,
            "reason": "raw_acquisition_only_v1",
        },
    }
    compile_control = _build_compile_request_payload(
        backend_name=(None if backend_name in {None, ""} else str(backend_name)),
        seed_transpiler=(None if seed_transpiler is None else int(seed_transpiler)),
        transpile_optimization_level=int(transpile_optimization_level),
        source="fixed_scaffold_runtime_transpile_cli",
    )
    compile_observation = _build_compile_observation_payload(
        requested=compile_control,
        backend_info=(
            main_eval.get("backend_info", {})
            if isinstance(main_eval.get("backend_info", {}), Mapping)
            else {}
        ),
    )
    backend_info_payload = (
        dict(main_eval.get("backend_info", {})) if isinstance(main_eval, Mapping) else {}
    )
    backend_details = (
        dict(backend_info_payload.get("details", {}))
        if isinstance(backend_info_payload.get("details", {}), Mapping)
        else {}
    )
    backend_details["symmetry_diagnostic"] = {
        "observable_family": symmetry_diagnostic.get("observable_family", None),
        "evaluation_id": symmetry_diagnostic.get("evaluation_id", None),
        "transport": symmetry_diagnostic.get("transport", None),
        "raw_artifact_path": symmetry_diagnostic.get("raw_artifact_path", None),
        "record_count": int(symmetry_diagnostic.get("record_count", 0) or 0),
        "diagonal_postprocessing_available": bool(
            diagonal_postprocessing.get("available", False)
        ),
        "group_count": symmetry_diagnostic.get("group_count", None),
        "term_count": symmetry_diagnostic.get("term_count", None),
        "compile_signatures_by_basis": dict(
            symmetry_diagnostic.get("compile_signatures_by_basis", {}) or {}
        ),
    }
    backend_info_payload["details"] = dict(backend_details)
    return {
        "success": True,
        "available": True,
        "route": "fixed_scaffold_runtime_raw_baseline",
        "artifact_json": str(ctx["path"]),
        "candidate_artifact_json": str(ctx["path"]),
        "source_kind": str(ctx.get("source_kind", "unknown")),
        "pool_type": str(subject.pool_type),
        "structure_locked": True,
        "reference_state_embedded": True,
        "ansatz_input_state_source": ansatz_input_state_meta.get("source", None),
        "ansatz_input_state_kind": ansatz_input_state_meta.get("handoff_state_kind", None),
        "noise_config": {
            "noise_mode": str(acquisition_noise_mode),
            "shots": int(shots),
            "oracle_repeats": int(oracle_repeats),
            "oracle_aggregate": str(oracle_aggregate),
            "backend_name": (None if backend_name in {None, ""} else str(backend_name)),
            "mitigation": dict(acquisition_mitigation_cfg),
            "symmetry_mitigation": dict(acquisition_symmetry_cfg),
            "runtime_profile": dict(runtime_profile_cfg),
            "runtime_session": dict(runtime_session_cfg),
            "execution_surface": "raw_measurement_v1",
            "raw_transport": str(raw_transport),
            "raw_store_memory": bool(raw_store_memory),
            "raw_artifact_path": (None if raw_artifact_path in {None, ""} else str(raw_artifact_path)),
            "requested_diagonal_postprocessing": {
                "mitigation": dict(requested_mitigation_cfg),
                "symmetry_mitigation": dict(effective_requested_symmetry_cfg),
                "order": "readout_then_symmetry",
                "observable_scope": "full_register_diagonal_only",
            },
        },
        "compile_control": dict(compile_control),
        "compile_observation": dict(compile_observation),
        "energy_audits": dict(phase_evals),
        "backend_info": dict(backend_info_payload),
        **_locked_subject_payload_fields(subject),
    }


def _run_saved_theta_local_mitigation_ablation(
    *,
    circuit: QuantumCircuit,
    artifact_json: str,
    subject: LockedImportedSubject,
    ordered_labels_exyz: Sequence[str],
    static_coeff_map_exyz: Mapping[str, complex],
    shots: int,
    seed: int,
    oracle_repeats: int,
    oracle_aggregate: str,
    backend_name: str | None,
    allow_aer_fallback: bool,
    omp_shm_workaround: bool,
    seed_transpiler: int | None,
    transpile_optimization_level: int,
    local_dd_probe_sequence: str | None,
) -> dict[str, Any]:
    phases: list[tuple[str, dict[str, Any]]] = [
        (
            "readout_only",
            normalize_mitigation_config(
                {
                    "mode": "readout",
                    "local_readout_strategy": "mthree",
                }
            ),
        ),
        (
            "readout_plus_gate_twirling",
            normalize_mitigation_config(
                {
                    "mode": "readout",
                    "local_readout_strategy": "mthree",
                    "local_gate_twirling": True,
                }
            ),
        ),
    ]
    if local_dd_probe_sequence not in {None, "", "none"}:
        phases.append(
            (
                "readout_plus_local_dd",
                normalize_mitigation_config(
                    {
                        "mode": "readout",
                        "local_readout_strategy": "mthree",
                        "dd_sequence": str(local_dd_probe_sequence),
                    }
                ),
            )
        )
        phases.append(
            (
                "readout_plus_gate_twirling_plus_local_dd",
                normalize_mitigation_config(
                    {
                        "mode": "readout",
                        "local_readout_strategy": "mthree",
                        "local_gate_twirling": True,
                        "dd_sequence": str(local_dd_probe_sequence),
                    }
                ),
            )
        )

    results: dict[str, Any] = {}
    for label, mitigation_cfg in phases:
        try:
            evaluation = _evaluate_locked_imported_circuit_energy(
                circuit=circuit,
                ordered_labels_exyz=ordered_labels_exyz,
                static_coeff_map_exyz=static_coeff_map_exyz,
                shots=int(shots),
                seed=int(seed),
                oracle_repeats=int(oracle_repeats),
                oracle_aggregate=str(oracle_aggregate),
                mitigation_config=dict(mitigation_cfg),
                symmetry_mitigation_config={"mode": "off"},
                backend_name=backend_name,
                use_fake_backend=True,
                allow_aer_fallback=bool(allow_aer_fallback),
                omp_shm_workaround=bool(omp_shm_workaround),
                seed_transpiler=(None if seed_transpiler is None else int(seed_transpiler)),
                transpile_optimization_level=int(transpile_optimization_level),
            )
            results[str(label)] = {
                "success": True,
                "available": True,
                "mitigation_config": dict(mitigation_cfg),
                **dict(evaluation),
            }
        except Exception as exc:
            results[str(label)] = {
                "success": False,
                "available": False,
                "reason": "saved_theta_local_probe_exception",
                "error": f"{type(exc).__name__}: {exc}",
                "mitigation_config": dict(mitigation_cfg),
            }

    return {
        "success": True,
        "available": True,
        "route": "fixed_scaffold_saved_theta_local_mitigation_ablation",
        "artifact_json": str(artifact_json),
        "theta_source": "imported_theta_runtime",
        "circuit_scope": "saved_theta_energy_probe",
        "execution_mode": "backend_scheduled",
        "backend_name": (None if backend_name is None else str(backend_name)),
        "subject_kind": str(subject.subject_kind),
        "term_order_id": str(subject.term_order_id),
        "operator_count": int(subject.operator_count),
        "runtime_term_count": int(subject.runtime_term_count),
        "run_config": {
            "shots": int(shots),
            "seed": int(seed),
            "oracle_repeats": int(oracle_repeats),
            "oracle_aggregate": str(oracle_aggregate),
            "seed_transpiler": (None if seed_transpiler is None else int(seed_transpiler)),
            "transpile_optimization_level": int(transpile_optimization_level),
        },
        "results": results,
    }


def _run_imported_fixed_scaffold_saved_theta_mitigation_matrix(
    *,
    artifact_json: str,
    shots: int,
    seed: int,
    oracle_repeats: int,
    oracle_aggregate: str,
    backend_name: str | None,
    use_fake_backend: bool,
    allow_aer_fallback: bool,
    omp_shm_workaround: bool,
    noise_mode: str = "backend_scheduled",
    compile_presets: Sequence[Mapping[str, Any]] = (),
    zne_scales: Sequence[float] = (1.0, 3.0, 5.0),
    suppression_labels: Sequence[str] = (
        "readout_plus_gate_twirling",
        "readout_plus_local_dd",
        "readout_plus_gate_twirling_plus_local_dd",
    ),
    selected_cells: Sequence[str] = (),
    mitigation_config_base: Mapping[str, Any] | None = None,
    symmetry_mitigation_config: Mapping[str, Any] | None = None,
    rank_policy: str = "delta_mean_then_two_qubit_then_depth_then_size",
    raw_artifact_root: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    ctx, ansatz_input_state_meta, subject, error_payload = _resolve_locked_imported_fixed_scaffold_context(
        artifact_json,
        nonfixed_reason="fixed_scaffold_saved_theta_mitigation_matrix_requires_locked_fixed_scaffold_source",
    )
    if error_payload is not None or ctx is None or subject is None:
        return dict(error_payload or {})

    if str(noise_mode).strip().lower() != "backend_scheduled":
        return {
            "success": False,
            "available": False,
            "reason": "fixed_scaffold_saved_theta_mitigation_matrix_requires_backend_scheduled",
            "artifact_json": str(ctx["path"]),
        }
    if not bool(use_fake_backend):
        return {
            "success": False,
            "available": False,
            "reason": "fixed_scaffold_saved_theta_mitigation_matrix_requires_local_fake_backend",
            "artifact_json": str(ctx["path"]),
        }
    backend_name_norm = None if backend_name in {None, "", "none"} else str(backend_name)
    if backend_name_norm is None:
        return {
            "success": False,
            "available": False,
            "reason": "fixed_scaffold_saved_theta_mitigation_matrix_requires_backend_name",
            "artifact_json": str(ctx["path"]),
        }
    mitigation_base_cfg = dict(
        normalize_mitigation_config(
            {
                "mode": "readout",
                "local_readout_strategy": "mthree",
                **dict(mitigation_config_base or {}),
            }
        )
    )
    symmetry_cfg = dict(
        normalize_symmetry_mitigation_config(
            {"mode": "off", **dict(symmetry_mitigation_config or {})}
        )
    )
    mitigation_base_mode = str(mitigation_base_cfg.get("mode", "none")).strip().lower()
    if mitigation_base_mode not in {"readout", "none"}:
        return {
            "success": False,
            "available": False,
            "reason": "fixed_scaffold_saved_theta_mitigation_matrix_unsupported_base_mitigation_mode",
            "artifact_json": str(ctx["path"]),
            "mitigation_mode": mitigation_base_mode,
        }
    if mitigation_base_mode == "readout" and str(
        mitigation_base_cfg.get("local_readout_strategy") or "mthree"
    ) != "mthree":
        return {
            "success": False,
            "available": False,
            "reason": "fixed_scaffold_saved_theta_mitigation_matrix_requires_mthree",
            "artifact_json": str(ctx["path"]),
        }
    mitigation_base_cfg["local_readout_strategy"] = (
        "mthree" if mitigation_base_mode == "readout" else None
    )
    def _actual_suppression_stack_label(mitigation_cfg: Mapping[str, Any]) -> str:
        gate_twirling = bool(mitigation_cfg.get("local_gate_twirling", False))
        has_dd = bool(mitigation_cfg.get("dd_sequence"))
        if mitigation_base_mode == "readout":
            if gate_twirling and has_dd:
                return "readout_plus_gate_twirling_plus_local_dd"
            if gate_twirling:
                return "readout_plus_gate_twirling"
            if has_dd:
                return "readout_plus_local_dd"
            return "readout_only"
        if gate_twirling and has_dd:
            return "gate_twirling_plus_local_dd"
        if gate_twirling:
            return "gate_twirling"
        if has_dd:
            return "local_dd"
        return "none"

    suppression_map: dict[str, tuple[str, dict[str, Any]]] = {
        "readout_plus_gate_twirling": (
            "twirl",
            normalize_mitigation_config(
                {
                    **dict(mitigation_base_cfg),
                    "local_gate_twirling": True,
                    "dd_sequence": None,
                }
            ),
        ),
        "readout_plus_local_dd": (
            "dd",
            normalize_mitigation_config(
                {
                    **dict(mitigation_base_cfg),
                    "local_gate_twirling": False,
                    "dd_sequence": "XpXm",
                }
            ),
        ),
        "readout_plus_gate_twirling_plus_local_dd": (
            "twirl_dd",
            normalize_mitigation_config(
                {
                    **dict(mitigation_base_cfg),
                    "local_gate_twirling": True,
                    "dd_sequence": "XpXm",
                }
            ),
        ),
    }
    normalized_suppression_labels = [str(x) for x in suppression_labels]
    unsupported_suppression = [x for x in normalized_suppression_labels if x not in suppression_map]
    if unsupported_suppression:
        return {
            "success": False,
            "available": False,
            "reason": "fixed_scaffold_saved_theta_mitigation_matrix_unsupported_suppression_stack",
            "artifact_json": str(ctx["path"]),
            "unsupported_suppression_labels": unsupported_suppression,
        }
    if not compile_presets:
        return {
            "success": False,
            "available": False,
            "reason": "fixed_scaffold_saved_theta_mitigation_matrix_requires_compile_presets",
            "artifact_json": str(ctx["path"]),
        }
    selected_cell_labels = [str(x).strip() for x in selected_cells if str(x).strip() != ""]
    artifact_compile_recommendation = _extract_fixed_scaffold_compile_recommendation(ctx)
    try:
        exact_energy_ref = (
            None
            if ctx.get("exact_energy", None) is None
            else float(ctx.get("exact_energy"))
            if math.isfinite(float(ctx.get("exact_energy")))
            else None
        )
    except Exception:
        exact_energy_ref = None

    def _emit_progress(event: str, **fields: Any) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(
                {
                    "event": str(event),
                    "route": "fixed_scaffold_saved_theta_mitigation_matrix",
                    "artifact_json": str(ctx["path"]),
                    **fields,
                }
            )
        except Exception:
            pass

    def _cell_counts_payload(cell_records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        successful = sum(1 for rec in cell_records if bool(rec.get("success", False)))
        return {
            "total": int(len(planned_cell_specs)),
            "completed": int(len(cell_records)),
            "successful": int(successful),
            "failed": int(len(cell_records) - successful),
        }

    def _best_by_group_from_records(
        records: Sequence[Mapping[str, Any]],
        *,
        key_name: str,
        key_values: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for rec in records:
            key_val = rec.get(key_name, None)
            if key_val in {None, ""}:
                continue
            grouped.setdefault(str(key_val), []).append(dict(rec))
        if key_values is not None:
            for key_val in key_values:
                grouped.setdefault(str(key_val), [])
        out: dict[str, Any] = {}
        for key_val, group in grouped.items():
            out[str(key_val)] = (dict(group[0]) if group else {})
        return out

    def _rank_cell_records(
        records: Sequence[Mapping[str, Any]],
        *,
        delta_field: str,
    ) -> list[dict[str, Any]]:
        filtered = [dict(rec) for rec in records if isinstance(rec, Mapping)]
        if str(delta_field) == "delta_to_exact_abs":
            filtered = [rec for rec in filtered if rec.get("delta_to_exact_abs", None) is not None]
        return (
            _rank_imported_compile_control_candidates(
                filtered,
                rank_policy=str(rank_policy),
                delta_field=str(delta_field),
            )
            if filtered
            else []
        )

    def _best_by_zne_toggle_from_records(
        records: Sequence[Mapping[str, Any]],
        *,
        delta_field: str,
    ) -> dict[str, Any]:
        zne_off_records = [rec for rec in records if not bool(rec.get("zne_enabled", False))]
        zne_on_records = [rec for rec in records if bool(rec.get("zne_enabled", False))]
        ranked_zne_off = _rank_cell_records(zne_off_records, delta_field=str(delta_field))
        ranked_zne_on = _rank_cell_records(zne_on_records, delta_field=str(delta_field))
        return {
            "zne_off": (dict(ranked_zne_off[0]) if ranked_zne_off else {}),
            "zne_on": (dict(ranked_zne_on[0]) if ranked_zne_on else {}),
        }

    def _partial_payload(
        *,
        reason: str,
        cell_records: Sequence[Mapping[str, Any]],
        last_cell_label: str | None,
        last_cell_index: int | None,
        elapsed_s: float,
    ) -> dict[str, Any]:
        completed_cells = [dict(rec) for rec in cell_records]
        successful_cells = [dict(rec) for rec in completed_cells if bool(rec.get("success", False))]
        ranked_cells_by_ideal_abs = _rank_cell_records(
            successful_cells,
            delta_field="delta_to_ideal_abs",
        )
        ranked_cells_by_exact_abs = _rank_cell_records(
            successful_cells,
            delta_field="delta_to_exact_abs",
        )
        payload: dict[str, Any] = {
            "success": False,
            "available": True,
            "route": "fixed_scaffold_saved_theta_mitigation_matrix",
            "reason": str(reason),
            "artifact_json": str(ctx["path"]),
            "source_kind": str(ctx.get("source_kind", "unknown")),
            "pool_type": str(subject.pool_type),
            "structure_locked": True,
            "theta_source": "imported_theta_runtime",
            "execution_mode": "backend_scheduled",
            "reference_state_embedded": True,
            "ansatz_input_state_source": ansatz_input_state_meta.get("source", None),
            "ansatz_input_state_kind": ansatz_input_state_meta.get("handoff_state_kind", None),
            "noise_config": {
                "noise_mode": "backend_scheduled",
                "shots": int(shots),
                "oracle_repeats": int(oracle_repeats),
                "oracle_aggregate": str(oracle_aggregate),
                "backend_name": str(backend_name_norm),
                "mitigation_base": dict(mitigation_base_cfg),
                "symmetry_mitigation": dict(symmetry_cfg),
                "raw_artifact_root": (
                    None if raw_artifact_root in {None, ""} else str(raw_artifact_root)
                ),
            },
            "ideal_reference": dict(ideal_eval),
            "exact_reference": {
                "exact_mean": exact_energy_ref,
                "exact_stderr": (0.0 if exact_energy_ref is not None else None),
            },
            "compile_presets": [dict(x) for x in compile_presets],
            "zne_scales": [float(x) for x in zne_scales],
            "suppression_labels": list(normalized_suppression_labels),
            "selected_cells": list(selected_cell_labels),
            "cells_partial": completed_cells,
            "cell_counts": _cell_counts_payload(completed_cells),
            "last_cell_label": (
                None if last_cell_label in {None, ""} else str(last_cell_label)
            ),
            "last_cell_index": (
                None if last_cell_index is None else int(last_cell_index)
            ),
            "best_cell": (
                dict(ranked_cells_by_ideal_abs[0]) if ranked_cells_by_ideal_abs else {}
            ),
            "best_cell_by_ideal_abs": (
                dict(ranked_cells_by_ideal_abs[0]) if ranked_cells_by_ideal_abs else {}
            ),
            "best_cell_by_exact_abs": (
                dict(ranked_cells_by_exact_abs[0]) if ranked_cells_by_exact_abs else {}
            ),
            "best_by_compile_preset": _best_by_group_from_records(
                ranked_cells_by_ideal_abs,
                key_name="compile_preset_label",
                key_values=[
                    str(x.get("label", ""))
                    for x in compile_presets
                    if str(x.get("label", "")).strip() != ""
                ],
            ),
            "best_by_compile_preset_by_ideal_abs": _best_by_group_from_records(
                ranked_cells_by_ideal_abs,
                key_name="compile_preset_label",
                key_values=[
                    str(x.get("label", ""))
                    for x in compile_presets
                    if str(x.get("label", "")).strip() != ""
                ],
            ),
            "best_by_compile_preset_by_exact_abs": _best_by_group_from_records(
                ranked_cells_by_exact_abs,
                key_name="compile_preset_label",
                key_values=[
                    str(x.get("label", ""))
                    for x in compile_presets
                    if str(x.get("label", "")).strip() != ""
                ],
            ),
            "best_by_zne_toggle": _best_by_zne_toggle_from_records(
                successful_cells,
                delta_field="delta_to_ideal_abs",
            ),
            "best_by_zne_toggle_by_ideal_abs": _best_by_zne_toggle_from_records(
                successful_cells,
                delta_field="delta_to_ideal_abs",
            ),
            "best_by_zne_toggle_by_exact_abs": _best_by_zne_toggle_from_records(
                successful_cells,
                delta_field="delta_to_exact_abs",
            ),
            "best_by_suppression_stack": _best_by_group_from_records(
                ranked_cells_by_ideal_abs,
                key_name="suppression_stack",
                key_values=list(normalized_suppression_labels),
            ),
            "best_by_suppression_stack_by_ideal_abs": _best_by_group_from_records(
                ranked_cells_by_ideal_abs,
                key_name="suppression_stack",
                key_values=list(normalized_suppression_labels),
            ),
            "best_by_suppression_stack_by_exact_abs": _best_by_group_from_records(
                ranked_cells_by_exact_abs,
                key_name="suppression_stack",
                key_values=list(normalized_suppression_labels),
            ),
            "artifact_compile_recommendation": (
                None
                if artifact_compile_recommendation is None
                else dict(artifact_compile_recommendation)
            ),
            "elapsed_s": float(elapsed_s),
            **_locked_subject_payload_fields(subject),
        }
        return payload

    nq = int(ctx["num_qubits"])
    theta_runtime = np.asarray(ctx["theta_runtime"], dtype=float).reshape(-1)
    ref_state = np.asarray(ctx["ansatz_input_state"], dtype=complex).reshape(-1)
    components = _build_locked_imported_fixed_theta_circuit_components(
        layout=ctx["layout"],
        theta_runtime=theta_runtime,
        nq=int(nq),
        ref_state=ref_state,
    )
    full_circuit = components["full_circuit"]
    reference_circuit = components["reference_circuit"]
    body_circuit = components["body_circuit"]
    qop = _build_locked_imported_energy_qop(
        ordered_labels_exyz=list(ctx["ordered_labels_exyz"]),
        static_coeff_map_exyz=dict(ctx["static_coeff_map_exyz"]),
    )
    ideal_eval = _evaluate_locked_imported_circuit_ideal_energy(
        circuit=full_circuit,
        ordered_labels_exyz=list(ctx["ordered_labels_exyz"]),
        static_coeff_map_exyz=dict(ctx["static_coeff_map_exyz"]),
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        qop=qop,
    )

    t0 = time.perf_counter()
    planned_cell_specs: list[dict[str, Any]] = []
    for preset in compile_presets:
        planned_preset_label = str(preset.get("label", "")).strip()
        if planned_preset_label == "":
            planned_preset_label = (
                f"opt{int(preset.get('transpile_optimization_level'))}"
                f"_seed{int(preset.get('seed_transpiler'))}"
            )
        for zne_enabled in (False, True):
            for suppression_label in normalized_suppression_labels:
                planned_cell_specs.append(
                    {
                        "preset_label": str(planned_preset_label),
                        "opt_level": int(preset.get("transpile_optimization_level")),
                        "seed_transpiler": int(preset.get("seed_transpiler")),
                        "zne_enabled": bool(zne_enabled),
                        "suppression_label": str(suppression_label),
                        "short_label": str(suppression_map[str(suppression_label)][0]),
                        "cell_label": (
                            f"{planned_preset_label}__zne_{'on' if bool(zne_enabled) else 'off'}"
                            f"__{suppression_map[str(suppression_label)][0]}"
                        ),
                    }
                )
    if selected_cell_labels:
        planned_label_set = {str(spec["cell_label"]) for spec in planned_cell_specs}
        unsupported_selected = [
            str(label) for label in selected_cell_labels if str(label) not in planned_label_set
        ]
        if unsupported_selected:
            return {
                "success": False,
                "available": False,
                "reason": "fixed_scaffold_saved_theta_mitigation_matrix_unsupported_selected_cells",
                "artifact_json": str(ctx["path"]),
                "unsupported_selected_cells": unsupported_selected,
            }
        selected_lookup = set(selected_cell_labels)
        planned_cell_specs = [
            dict(spec) for spec in planned_cell_specs if str(spec["cell_label"]) in selected_lookup
        ]
    planned_cell_labels = [str(spec["cell_label"]) for spec in planned_cell_specs]
    _ai_log(
        "fixed_scaffold_saved_theta_mitigation_matrix_initialized",
        artifact_json=str(ctx["path"]),
        cell_total=int(len(planned_cell_specs)),
        ideal_mean=float(ideal_eval["ideal_mean"]),
        exact_mean=exact_energy_ref,
    )
    _emit_progress(
        "fixed_scaffold_saved_theta_mitigation_matrix_initialized",
        cell_counts=_cell_counts_payload(()),
        cell_labels=list(planned_cell_labels),
        ideal_mean=float(ideal_eval["ideal_mean"]),
        ideal_stderr=float(ideal_eval["ideal_stderr"]),
        exact_mean=exact_energy_ref,
        exact_stderr=(0.0 if exact_energy_ref is not None else None),
        artifact_compile_recommendation=(
            None
            if artifact_compile_recommendation is None
            else dict(artifact_compile_recommendation)
        ),
        elapsed_s=float(time.perf_counter() - t0),
        partial_payload=_partial_payload(
            reason="in_progress",
            cell_records=(),
            last_cell_label=None,
            last_cell_index=None,
            elapsed_s=float(time.perf_counter() - t0),
        ),
    )
    cells: list[dict[str, Any]] = []
    for cell_index, spec in enumerate(planned_cell_specs):
        preset_label = str(spec["preset_label"])
        opt_level = int(spec["opt_level"])
        seed_transpiler = int(spec["seed_transpiler"])
        zne_enabled = bool(spec["zne_enabled"])
        suppression_label = str(spec["suppression_label"])
        short_label = str(spec["short_label"])
        cell_label = str(spec["cell_label"])
        mitigation_cfg = suppression_map[str(suppression_label)][1]
        actual_suppression_stack = _actual_suppression_stack_label(mitigation_cfg)
        compile_request = _build_compile_request_payload(
            backend_name=backend_name_norm,
            seed_transpiler=int(seed_transpiler),
            transpile_optimization_level=int(opt_level),
            source="fixed_scaffold_saved_theta_mitigation_matrix_cell",
        )
        _ai_log(
            "fixed_scaffold_saved_theta_mitigation_matrix_cell_start",
            cell_index=int(cell_index),
            cell_label=str(cell_label),
            compile_preset_label=str(preset_label),
            suppression_stack=str(actual_suppression_stack),
            zne_enabled=bool(zne_enabled),
            seed_transpiler=int(seed_transpiler),
            transpile_optimization_level=int(opt_level),
            cell_total=int(len(planned_cell_specs)),
        )
        _emit_progress(
            "fixed_scaffold_saved_theta_mitigation_matrix_cell_started",
            cell_index=int(cell_index),
            cell_label=str(cell_label),
            compile_request=dict(compile_request),
            cell_counts=_cell_counts_payload(cells),
            elapsed_s=float(time.perf_counter() - t0),
            partial_payload=_partial_payload(
                reason="in_progress",
                cell_records=cells,
                last_cell_label=str(cell_label),
                last_cell_index=int(cell_index),
                elapsed_s=float(time.perf_counter() - t0),
            ),
        )
        try:
            if bool(zne_enabled):
                cell_eval = _evaluate_locked_imported_circuit_energy_local_zne(
                    reference_circuit=reference_circuit,
                    body_circuit=body_circuit,
                    ordered_labels_exyz=list(ctx["ordered_labels_exyz"]),
                    static_coeff_map_exyz=dict(ctx["static_coeff_map_exyz"]),
                    shots=int(shots),
                    seed=int(seed),
                    oracle_repeats=int(oracle_repeats),
                    oracle_aggregate=str(oracle_aggregate),
                    mitigation_config=dict(mitigation_cfg),
                    symmetry_mitigation_config=dict(symmetry_cfg),
                    backend_name=backend_name_norm,
                    use_fake_backend=True,
                    allow_aer_fallback=bool(allow_aer_fallback),
                    omp_shm_workaround=bool(omp_shm_workaround),
                    seed_transpiler=int(seed_transpiler),
                    transpile_optimization_level=int(opt_level),
                    zne_scales=list(zne_scales),
                    ideal_eval=ideal_eval,
                    exact_energy=exact_energy_ref,
                    qop=qop,
                    compile_request_source="fixed_scaffold_saved_theta_mitigation_matrix_cell",
                )
            else:
                noisy_eval = _evaluate_locked_imported_circuit_noisy_energy(
                    circuit=full_circuit,
                    ordered_labels_exyz=list(ctx["ordered_labels_exyz"]),
                    static_coeff_map_exyz=dict(ctx["static_coeff_map_exyz"]),
                    shots=int(shots),
                    seed=int(seed),
                    oracle_repeats=int(oracle_repeats),
                    oracle_aggregate=str(oracle_aggregate),
                    mitigation_config=dict(mitigation_cfg),
                    symmetry_mitigation_config=dict(symmetry_cfg),
                    backend_name=backend_name_norm,
                    use_fake_backend=True,
                    allow_aer_fallback=bool(allow_aer_fallback),
                    omp_shm_workaround=bool(omp_shm_workaround),
                    seed_transpiler=int(seed_transpiler),
                    transpile_optimization_level=int(opt_level),
                    qop=qop,
                )
                combined = _combine_locked_imported_circuit_energy_evaluations(
                    noisy_eval=noisy_eval,
                    ideal_eval=ideal_eval,
                    exact_energy=exact_energy_ref,
                )
                backend_info = (
                    dict(combined.get("backend_info", {}))
                    if isinstance(combined.get("backend_info", {}), Mapping)
                    else {}
                )
                compile_observation = _build_compile_observation_payload(
                    requested=compile_request,
                    backend_info=backend_info,
                )
                compile_metrics = _extract_compile_metrics_from_backend_info(backend_info)
                cell_eval = {
                    "success": True,
                    "zne_enabled": False,
                    "zne_scales": [],
                    "per_factor_results": [],
                    "extrapolator": None,
                    "compile_request": dict(compile_request),
                    "compile_observation": dict(compile_observation),
                    "matches_requested": compile_observation.get("matches_requested", None),
                    "transpile_seed": compile_metrics.get("transpile_seed", None),
                    "transpile_optimization_level": compile_metrics.get(
                        "transpile_optimization_level", None
                    ),
                    "compiled_two_qubit_count": int(
                        compile_metrics.get("compiled_two_qubit_count", 0)
                    ),
                    "compiled_depth": int(compile_metrics.get("compiled_depth", 0)),
                    "compiled_size": int(compile_metrics.get("compiled_size", 0)),
                    "backend_info": dict(backend_info),
                    **dict(combined),
                }
            compile_observation = (
                dict(cell_eval.get("compile_observation", {}))
                if isinstance(cell_eval.get("compile_observation", {}), Mapping)
                else {}
            )
            cell_noisy_mean = float(cell_eval.get("noisy_mean", float("nan")))
            cell_noisy_stderr = float(cell_eval.get("noisy_stderr", float("nan")))
            cell_ideal_mean = float(cell_eval.get("ideal_mean", float("nan")))
            cell_ideal_stderr = float(cell_eval.get("ideal_stderr", float("nan")))
            delta_to_ideal_mean = cell_eval.get(
                "delta_to_ideal_mean",
                cell_eval.get("delta_mean", None),
            )
            delta_to_ideal_stderr = cell_eval.get(
                "delta_to_ideal_stderr",
                cell_eval.get("delta_stderr", None),
            )
            delta_to_ideal_abs = cell_eval.get(
                "delta_to_ideal_abs",
                None
                if delta_to_ideal_mean is None
                else float(abs(float(delta_to_ideal_mean))),
            )
            exact_mean = cell_eval.get("exact_mean", exact_energy_ref)
            exact_stderr = cell_eval.get(
                "exact_stderr",
                (0.0 if exact_mean is not None else None),
            )
            delta_to_exact_mean = cell_eval.get("delta_to_exact_mean", None)
            delta_to_exact_stderr = cell_eval.get("delta_to_exact_stderr", None)
            delta_to_exact_abs = cell_eval.get("delta_to_exact_abs", None)
            if (
                delta_to_exact_mean is None
                and exact_mean is not None
                and math.isfinite(cell_noisy_mean)
                and math.isfinite(cell_noisy_stderr)
            ):
                delta_to_exact_mean = float(cell_noisy_mean - float(exact_mean))
                delta_to_exact_stderr = float(_combine_stderr(cell_noisy_stderr, 0.0))
                delta_to_exact_abs = float(abs(delta_to_exact_mean))
            cells.append(
                {
                    "success": True,
                    "cell_index": int(cell_index),
                    "label": str(cell_label),
                    "compile_preset_label": str(preset_label),
                    "seed_transpiler": int(seed_transpiler),
                    "transpile_optimization_level": int(opt_level),
                    "zne_enabled": bool(zne_enabled),
                    "suppression_stack": str(actual_suppression_stack),
                    "suppression_stack_label": str(short_label),
                    "mitigation_config": dict(mitigation_cfg),
                    "compile_request": dict(cell_eval.get("compile_request", compile_request)),
                    "compile_observation": compile_observation,
                    "matches_requested": cell_eval.get("matches_requested", None),
                    "transpile_seed": cell_eval.get("transpile_seed", None),
                    "compiled_two_qubit_count": int(
                        cell_eval.get("compiled_two_qubit_count", 0)
                    ),
                    "compiled_depth": int(cell_eval.get("compiled_depth", 0)),
                    "compiled_size": int(cell_eval.get("compiled_size", 0)),
                    "noisy_mean": float(cell_noisy_mean),
                    "noisy_stderr": float(cell_noisy_stderr),
                    "energy_noisy_mean": float(cell_noisy_mean),
                    "energy_noisy_stderr": float(cell_noisy_stderr),
                    "ideal_mean": float(cell_ideal_mean),
                    "ideal_stderr": float(cell_ideal_stderr),
                    "energy_ideal_mean": float(cell_ideal_mean),
                    "energy_ideal_stderr": float(cell_ideal_stderr),
                    "exact_mean": (
                        None if exact_mean is None else float(exact_mean)
                    ),
                    "exact_stderr": (
                        None if exact_stderr is None else float(exact_stderr)
                    ),
                    "energy_exact_mean": (
                        None if exact_mean is None else float(exact_mean)
                    ),
                    "energy_exact_stderr": (
                        None if exact_stderr is None else float(exact_stderr)
                    ),
                    "delta_mean": (
                        float(delta_to_ideal_mean)
                        if delta_to_ideal_mean is not None
                        else float("nan")
                    ),
                    "delta_stderr": (
                        float(delta_to_ideal_stderr)
                        if delta_to_ideal_stderr is not None
                        else float("nan")
                    ),
                    "delta_abs": (
                        float(delta_to_ideal_abs)
                        if delta_to_ideal_abs is not None
                        else float("nan")
                    ),
                    "delta_to_ideal_mean": (
                        float(delta_to_ideal_mean)
                        if delta_to_ideal_mean is not None
                        else float("nan")
                    ),
                    "delta_to_ideal_stderr": (
                        float(delta_to_ideal_stderr)
                        if delta_to_ideal_stderr is not None
                        else float("nan")
                    ),
                    "delta_to_ideal_abs": (
                        float(delta_to_ideal_abs)
                        if delta_to_ideal_abs is not None
                        else float("nan")
                    ),
                    "delta_to_exact_mean": (
                        None if delta_to_exact_mean is None else float(delta_to_exact_mean)
                    ),
                    "delta_to_exact_stderr": (
                        None if delta_to_exact_stderr is None else float(delta_to_exact_stderr)
                    ),
                    "delta_to_exact_abs": (
                        None if delta_to_exact_abs is None else float(delta_to_exact_abs)
                    ),
                    "backend_info": dict(cell_eval.get("backend_info", {})),
                    "zne_scales": list(cell_eval.get("zne_scales", [])),
                    "per_factor_results": list(cell_eval.get("per_factor_results", [])),
                    "extrapolator": cell_eval.get("extrapolator", None),
                    "extrapolated_energy_mean": cell_eval.get(
                        "extrapolated_energy_mean", None
                    ),
                    "extrapolated_delta_mean": cell_eval.get("delta_mean", None),
                    "zne_fold_scope": cell_eval.get("zne_fold_scope", None),
                    "zne_fold_warning": cell_eval.get("zne_fold_warning", None),
                    "folding_metadata": (
                        dict(cell_eval.get("folding_metadata", {}))
                        if isinstance(cell_eval.get("folding_metadata", {}), Mapping)
                        else {}
                    ),
                }
            )
        except Exception as exc:
            cells.append(
                {
                    "success": False,
                    "cell_index": int(cell_index),
                    "label": str(cell_label),
                    "compile_preset_label": str(preset_label),
                    "seed_transpiler": int(seed_transpiler),
                    "transpile_optimization_level": int(opt_level),
                    "zne_enabled": bool(zne_enabled),
                    "suppression_stack": str(actual_suppression_stack),
                    "suppression_stack_label": str(short_label),
                    "mitigation_config": dict(mitigation_cfg),
                    "compile_request": dict(compile_request),
                    "compile_observation": {
                        "available": False,
                        "requested": dict(compile_request),
                        "observed": None,
                        "matches_requested": None,
                        "mismatch_fields": [],
                        "reason": "cell_exception",
                    },
                    "reason": "cell_exception",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        last_cell = dict(cells[-1])
        _ai_log(
            "fixed_scaffold_saved_theta_mitigation_matrix_cell_done",
            cell_index=int(cell_index),
            cell_label=str(cell_label),
            success=bool(last_cell.get("success", False)),
            compile_preset_label=str(preset_label),
            suppression_stack=str(actual_suppression_stack),
            zne_enabled=bool(zne_enabled),
            delta_mean=last_cell.get("delta_mean", None),
            delta_to_ideal_mean=last_cell.get("delta_to_ideal_mean", None),
            delta_to_exact_mean=last_cell.get("delta_to_exact_mean", None),
            delta_stderr=last_cell.get("delta_stderr", None),
            compiled_two_qubit_count=last_cell.get("compiled_two_qubit_count", None),
            compiled_depth=last_cell.get("compiled_depth", None),
            compiled_size=last_cell.get("compiled_size", None),
            elapsed_s=float(time.perf_counter() - t0),
        )
        _emit_progress(
            "fixed_scaffold_saved_theta_mitigation_matrix_cell_completed",
            cell_index=int(cell_index),
            cell_label=str(cell_label),
            cell=last_cell,
            cell_counts=_cell_counts_payload(cells),
            elapsed_s=float(time.perf_counter() - t0),
            partial_payload=_partial_payload(
                reason="in_progress",
                cell_records=cells,
                last_cell_label=str(cell_label),
                last_cell_index=int(cell_index),
                elapsed_s=float(time.perf_counter() - t0),
            ),
        )

    successful_cells = [dict(rec) for rec in cells if bool(rec.get("success", False))]
    ranked_cells_by_ideal_abs = _rank_cell_records(
        successful_cells,
        delta_field="delta_to_ideal_abs",
    )
    ranked_cells_by_exact_abs = _rank_cell_records(
        successful_cells,
        delta_field="delta_to_exact_abs",
    )

    failed_count = int(sum(1 for rec in cells if not bool(rec.get("success", False))))
    payload: dict[str, Any] = {
        "success": bool(failed_count == 0 and len(cells) > 0),
        "available": True,
        "route": "fixed_scaffold_saved_theta_mitigation_matrix",
        "artifact_json": str(ctx["path"]),
        "source_kind": str(ctx.get("source_kind", "unknown")),
        "pool_type": str(subject.pool_type),
        "structure_locked": True,
        "theta_source": "imported_theta_runtime",
        "execution_mode": "backend_scheduled",
        "reference_state_embedded": True,
        "ansatz_input_state_source": ansatz_input_state_meta.get("source", None),
        "ansatz_input_state_kind": ansatz_input_state_meta.get("handoff_state_kind", None),
        "noise_config": {
            "noise_mode": "backend_scheduled",
            "shots": int(shots),
            "oracle_repeats": int(oracle_repeats),
            "oracle_aggregate": str(oracle_aggregate),
            "backend_name": str(backend_name_norm),
            "mitigation_base": dict(mitigation_base_cfg),
            "symmetry_mitigation": dict(symmetry_cfg),
            "raw_artifact_root": (None if raw_artifact_root in {None, ""} else str(raw_artifact_root)),
        },
        "ideal_reference": dict(ideal_eval),
        "exact_reference": {
            "exact_mean": exact_energy_ref,
            "exact_stderr": (0.0 if exact_energy_ref is not None else None),
        },
        "compile_presets": [dict(x) for x in compile_presets],
        "zne_scales": [float(x) for x in zne_scales],
        "suppression_labels": list(normalized_suppression_labels),
        "selected_cells": list(selected_cell_labels),
        "cells": cells,
        "cell_counts": _cell_counts_payload(cells),
        "best_cell": (
            dict(ranked_cells_by_ideal_abs[0]) if ranked_cells_by_ideal_abs else {}
        ),
        "best_cell_by_ideal_abs": (
            dict(ranked_cells_by_ideal_abs[0]) if ranked_cells_by_ideal_abs else {}
        ),
        "best_cell_by_exact_abs": (
            dict(ranked_cells_by_exact_abs[0]) if ranked_cells_by_exact_abs else {}
        ),
        "best_by_compile_preset": _best_by_group_from_records(
            ranked_cells_by_ideal_abs,
            key_name="compile_preset_label",
            key_values=[
                str(x.get("label", ""))
                for x in compile_presets
                if str(x.get("label", "")).strip() != ""
            ],
        ),
        "best_by_compile_preset_by_ideal_abs": _best_by_group_from_records(
            ranked_cells_by_ideal_abs,
            key_name="compile_preset_label",
            key_values=[
                str(x.get("label", ""))
                for x in compile_presets
                if str(x.get("label", "")).strip() != ""
            ],
        ),
        "best_by_compile_preset_by_exact_abs": _best_by_group_from_records(
            ranked_cells_by_exact_abs,
            key_name="compile_preset_label",
            key_values=[
                str(x.get("label", ""))
                for x in compile_presets
                if str(x.get("label", "")).strip() != ""
            ],
        ),
        "best_by_zne_toggle": _best_by_zne_toggle_from_records(
            successful_cells,
            delta_field="delta_to_ideal_abs",
        ),
        "best_by_zne_toggle_by_ideal_abs": _best_by_zne_toggle_from_records(
            successful_cells,
            delta_field="delta_to_ideal_abs",
        ),
        "best_by_zne_toggle_by_exact_abs": _best_by_zne_toggle_from_records(
            successful_cells,
            delta_field="delta_to_exact_abs",
        ),
        "best_by_suppression_stack": _best_by_group_from_records(
            ranked_cells_by_ideal_abs,
            key_name="suppression_stack",
            key_values=list(normalized_suppression_labels),
        ),
        "best_by_suppression_stack_by_ideal_abs": _best_by_group_from_records(
            ranked_cells_by_ideal_abs,
            key_name="suppression_stack",
            key_values=list(normalized_suppression_labels),
        ),
        "best_by_suppression_stack_by_exact_abs": _best_by_group_from_records(
            ranked_cells_by_exact_abs,
            key_name="suppression_stack",
            key_values=list(normalized_suppression_labels),
        ),
        "artifact_compile_recommendation": (
            None
            if artifact_compile_recommendation is None
            else dict(artifact_compile_recommendation)
        ),
        "elapsed_s": float(time.perf_counter() - t0),
        **_locked_subject_payload_fields(subject),
    }
    if not bool(payload["success"]):
        payload["reason"] = (
            "one_or_more_cells_failed"
            if len(cells) > 0
            else "fixed_scaffold_saved_theta_mitigation_matrix_no_cells"
        )
    return payload


def _run_imported_fixed_scaffold_noisy_replay(
    *,
    artifact_json: str,
    shots: int,
    seed: int,
    oracle_repeats: int,
    oracle_aggregate: str,
    backend_name: str | None,
    use_fake_backend: bool,
    allow_aer_fallback: bool,
    omp_shm_workaround: bool,
    optimizer_method: str,
    optimizer_seed: int,
    optimizer_maxiter: int,
    optimizer_wallclock_cap_s: int,
    spsa_a: float,
    spsa_c: float,
    spsa_alpha: float,
    spsa_gamma: float,
    spsa_A: float,
    spsa_avg_last: int,
    spsa_eval_repeats: int,
    spsa_eval_agg: str,
    mitigation_config: dict[str, Any],
    symmetry_mitigation_config: dict[str, Any],
    runtime_profile_config: dict[str, Any] | None = None,
    runtime_session_config: dict[str, Any] | None = None,
    transpile_optimization_level: int = 1,
    seed_transpiler: int | None = None,
    include_dd_probe: bool = False,
    include_final_zne_audit: bool = False,
    progress_every_s: float = 60.0,
    local_dd_probe_sequence: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    ctx, ansatz_input_state_meta, subject, error_payload = _resolve_locked_imported_fixed_scaffold_context(
        artifact_json,
        nonfixed_reason="fixed_scaffold_replay_requires_locked_fixed_scaffold_source",
    )
    if error_payload is not None or ctx is None or subject is None:
        return dict(error_payload or {})
    if not bool(use_fake_backend):
        payload = {
            "success": False,
            "available": False,
            "reason": "fixed_scaffold_replay_local_fake_backend_required",
            "artifact_json": str(ctx["path"]),
            "structure_locked": True,
            "pool_type": str(subject.pool_type),
            **_locked_subject_payload_fields(subject),
        }
        _ai_log(
            "fixed_scaffold_replay_failed",
            route="fixed_scaffold_noisy_replay",
            reason=str(payload["reason"]),
            artifact_json=str(ctx["path"]),
            subject_kind=str(subject.subject_kind),
        )
        return payload

    mitigation_cfg = dict(normalize_mitigation_config(mitigation_config))
    symmetry_cfg = dict(normalize_symmetry_mitigation_config(symmetry_mitigation_config))
    mitigation_mode_key = str(mitigation_cfg.get("mode", "none")).strip().lower()
    symmetry_mode_key = str(symmetry_cfg.get("mode", "off")).strip().lower()
    if bool(use_fake_backend):
        if mitigation_mode_key not in {"none", "readout"}:
            return {
                "success": False,
                "available": False,
                "reason": "fixed_scaffold_replay_backend_scheduled_supports_only_none_or_readout",
                "artifact_json": str(ctx["path"]),
            }
        if mitigation_mode_key == "readout":
            strategy = str(mitigation_cfg.get("local_readout_strategy") or "mthree")
            if strategy != "mthree":
                return {
                    "success": False,
                    "available": False,
                    "reason": "fixed_scaffold_replay_requires_mthree_for_backend_scheduled",
                    "artifact_json": str(ctx["path"]),
                }
            mitigation_cfg["local_readout_strategy"] = "mthree"
        else:
            mitigation_cfg["local_readout_strategy"] = None
    else:
        mitigation_cfg["local_readout_strategy"] = None
    if bool(mitigation_cfg.get("dd_sequence")):
        return {
            "success": False,
            "available": False,
            "reason": "fixed_scaffold_replay_keeps_dd_out_of_optimizer_loop",
            "artifact_json": str(ctx["path"]),
        }
    optimizer_method_key = str(optimizer_method).strip().lower()
    if optimizer_method_key not in {"spsa", "powell"}:
        return {
            "success": False,
            "available": False,
            "reason": "fixed_scaffold_replay_requires_spsa_or_powell",
            "artifact_json": str(ctx["path"]),
            "optimizer_method": str(optimizer_method),
        }

    layout = ctx["layout"]
    nq = int(ctx["num_qubits"])
    theta0 = np.asarray(ctx["theta_runtime"], dtype=float).reshape(-1)
    ref_state = np.asarray(ctx["ansatz_input_state"], dtype=complex).reshape(-1)
    ordered_labels_exyz = list(ctx["ordered_labels_exyz"])
    static_coeff_map_exyz = dict(ctx["static_coeff_map_exyz"])
    qop = build_time_dependent_sparse_qop(
        ordered_labels_exyz=ordered_labels_exyz,
        static_coeff_map_exyz=static_coeff_map_exyz,
        drive_coeff_map_exyz=None,
    )
    theta0_logical = np.asarray(project_runtime_theta_block_mean(theta0, layout), dtype=float).reshape(-1)

    noisy_mode = "backend_scheduled"
    if mitigation_mode_key == "readout":
        local_mitigation_label = (
            "readout_plus_gate_twirling"
            if bool(mitigation_cfg.get("local_gate_twirling", False))
            else "readout_only"
        )
    elif mitigation_mode_key == "none":
        local_mitigation_label = (
            "gate_twirling_only"
            if bool(mitigation_cfg.get("local_gate_twirling", False))
            else "none"
        )
    else:
        local_mitigation_label = str(mitigation_mode_key)
    runtime_profile_cfg = dict(normalize_runtime_estimator_profile_config(runtime_profile_config))
    runtime_session_cfg = dict(normalize_runtime_session_policy_config(runtime_session_config))
    noisy_cfg = OracleConfig(
        noise_mode=str(noisy_mode),
        shots=int(shots),
        seed=int(seed),
        seed_transpiler=(None if seed_transpiler is None else int(seed_transpiler)),
        transpile_optimization_level=int(transpile_optimization_level),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        backend_name=(None if backend_name is None else str(backend_name)),
        use_fake_backend=bool(use_fake_backend),
        allow_aer_fallback=bool(allow_aer_fallback),
        aer_fallback_mode="sampler_shots",
        omp_shm_workaround=bool(omp_shm_workaround),
        mitigation=dict(mitigation_cfg),
        symmetry_mitigation=dict(symmetry_cfg),
        runtime_profile=dict(runtime_profile_cfg),
        runtime_session=dict(runtime_session_cfg),
    )

    class _WallclockStop(RuntimeError):
        pass

    t0 = time.perf_counter()
    objective_history_tail: list[dict[str, Any]] = []
    objective_trace: list[dict[str, Any]] = []
    runtime_job_ids: list[str] = []
    objective_calls_total = 0
    best_eval: dict[str, Any] | None = None
    wallclock_hit = False
    optimizer_exception: Exception | None = None
    res = None
    progress_interval_s = float(max(1.0, progress_every_s))
    last_progress_emit_s = -progress_interval_s
    last_spsa_payload: dict[str, Any] | None = None

    def _emit_replay_log(event: str, **fields: Any) -> None:
        try:
            _ai_log(
                event,
                route="fixed_scaffold_noisy_replay",
                artifact_json=str(ctx["path"]),
                subject_kind=str(subject.subject_kind),
                theta_source="imported_theta_runtime",
                execution_mode="backend_scheduled",
                backend_name=(None if backend_name is None else str(backend_name)),
                local_mitigation_label=str(local_mitigation_label),
                symmetry_mitigation_mode=str(symmetry_mode_key),
                **fields,
            )
        except Exception:
            return

    def _emit_progress(event: str, **fields: Any) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(
                {
                    "event": str(event),
                    "route": "fixed_scaffold_noisy_replay",
                    "artifact_json": str(ctx["path"]),
                    **fields,
                }
            )
        except Exception:
            pass

    def _current_backend_info_payload(oracle: Any | None = None) -> dict[str, Any]:
        info = getattr(oracle, "backend_info", None)
        details = {}
        if info is not None:
            raw_details = getattr(info, "details", {})
            if isinstance(raw_details, Mapping):
                details = dict(raw_details)
            return {
                "noise_mode": str(getattr(info, "noise_mode", noisy_mode)),
                "estimator_kind": str(getattr(info, "estimator_kind", "expectation_oracle")),
                "backend_name": getattr(info, "backend_name", backend_name),
                "using_fake_backend": bool(getattr(info, "using_fake_backend", use_fake_backend)),
                "details": details,
            }
        return {
            "noise_mode": str(noisy_mode),
            "estimator_kind": "expectation_oracle",
            "backend_name": backend_name,
            "using_fake_backend": bool(use_fake_backend),
            "details": {},
        }

    def _best_so_far_payload() -> dict[str, Any]:
        if best_eval is None:
            return {}
        theta_best_runtime = np.asarray(best_eval["theta_runtime"], dtype=float).reshape(-1)
        theta_best_logical = np.asarray(
            project_runtime_theta_block_mean(theta_best_runtime, layout),
            dtype=float,
        ).reshape(-1)
        return {
            "call_index": int(best_eval["call_index"]),
            "energy_noisy_mean": float(best_eval["energy_noisy_mean"]),
            "energy_noisy_stderr": float(best_eval["energy_noisy_stderr"]),
            "theta_runtime": [float(x) for x in theta_best_runtime.tolist()],
            "theta_logical": [float(x) for x in theta_best_logical.tolist()],
        }

    def _partial_payload(
        *,
        reason: str,
        current_backend_info: Mapping[str, Any] | None = None,
        success: bool = False,
        partial: bool = True,
        optimizer_stop_reason: str = "in_progress",
        optimizer_message: str = "in_progress",
    ) -> dict[str, Any]:
        best_so_far = _best_so_far_payload()
        theta_payload: dict[str, Any] = {
            "initial_runtime": [float(x) for x in theta0.tolist()],
            "initial_logical": [float(x) for x in theta0_logical.tolist()],
        }
        if best_so_far:
            theta_payload["best_runtime"] = list(best_so_far["theta_runtime"])
            theta_payload["best_logical"] = list(best_so_far["theta_logical"])
        energies: dict[str, Any] = {}
        try:
            saved_energy_raw = ctx.get("saved_energy", None)
            if saved_energy_raw is not None and math.isfinite(float(saved_energy_raw)):
                energies["saved_artifact_energy"] = float(saved_energy_raw)
        except Exception:
            pass
        if best_so_far:
            energies["best_noisy_mean"] = float(best_so_far["energy_noisy_mean"])
            energies["best_noisy_stderr"] = float(best_so_far["energy_noisy_stderr"])
        optimizer_iterations_completed = 0
        if isinstance(last_spsa_payload, Mapping):
            try:
                optimizer_iterations_completed = int(last_spsa_payload.get("iter", 0) or 0)
            except Exception:
                optimizer_iterations_completed = 0
        payload: dict[str, Any] = {
            "success": bool(success),
            "available": True,
            "partial": bool(partial),
            "route": "fixed_scaffold_noisy_replay",
            "reason": str(reason),
            "artifact_json": str(ctx["path"]),
            "source_kind": str(ctx.get("source_kind", "unknown")),
            "pool_type": str(subject.pool_type),
            "structure_locked": True,
            "matched_family_replay": False,
            "full_circuit_import_audit": False,
            "reps": 1,
            "theta_source": "imported_theta_runtime",
            "execution_mode": "backend_scheduled",
            "local_mitigation_label": str(local_mitigation_label),
            "reference_state_embedded": True,
            "ansatz_input_state_source": ansatz_input_state_meta.get("source", None),
            "ansatz_input_state_kind": ansatz_input_state_meta.get("handoff_state_kind", None),
            "parameterization": {
                "mode": "per_pauli_term_v1",
                "logical_parameter_count": int(layout.logical_parameter_count),
                "runtime_parameter_count": int(layout.runtime_parameter_count),
            },
            "optimizer": {
                "method": ("SPSA" if optimizer_method_key == "spsa" else "Powell"),
                "seed": int(optimizer_seed),
                "maxiter": int(optimizer_maxiter),
                "wallclock_cap_s": int(optimizer_wallclock_cap_s),
                "stop_reason": str(optimizer_stop_reason),
                "iterations_completed": int(optimizer_iterations_completed),
                "objective_calls_total": int(objective_calls_total),
                "message": str(optimizer_message),
            },
            "noise_config": {
                "noise_mode": str(noisy_mode),
                "shots": int(shots),
                "oracle_repeats": int(oracle_repeats),
                "oracle_aggregate": str(oracle_aggregate),
                "mitigation": dict(mitigation_cfg),
                "symmetry_mitigation": dict(symmetry_cfg),
                "runtime_profile": dict(runtime_profile_cfg),
                "runtime_session": dict(runtime_session_cfg),
            },
            "compile_control": {
                "transpile_optimization_level": int(transpile_optimization_level),
                "seed_transpiler": (
                    None if seed_transpiler is None else int(seed_transpiler)
                ),
                "source": "fake_backend_cli",
            },
            "theta": theta_payload,
            "objective_history_tail": [dict(row) for row in objective_history_tail],
            "objective_trace": [dict(row) for row in objective_trace],
            "runtime_job_ids": [str(x) for x in runtime_job_ids],
            "backend_info": dict(current_backend_info or _current_backend_info_payload()),
            "elapsed_s": float(time.perf_counter() - t0),
            **_locked_subject_payload_fields(subject),
        }
        if energies:
            payload["energies"] = energies
        if best_so_far:
            payload["best_so_far"] = dict(best_so_far)
        return payload

    def _maybe_emit_progress(
        *,
        call_index: int,
        elapsed_s: float,
        current_energy: float | None = None,
    ) -> None:
        nonlocal last_progress_emit_s
        if float(elapsed_s) < float(last_progress_emit_s + progress_interval_s):
            return
        payload: dict[str, Any] = {
            "call_index": int(call_index),
            "elapsed_s": float(elapsed_s),
            "best_call_index": (None if best_eval is None else int(best_eval["call_index"])),
            "best_energy_noisy_mean": (
                None if best_eval is None else float(best_eval["energy_noisy_mean"])
            ),
            "current_energy_noisy_mean": (None if current_energy is None else float(current_energy)),
        }
        if isinstance(last_spsa_payload, Mapping):
            payload.update(
                {
                    "iter": int(last_spsa_payload.get("iter", 0)),
                    "nfev_so_far": int(last_spsa_payload.get("nfev_so_far", 0)),
                    "grad_norm": float(last_spsa_payload.get("grad_norm", float("nan"))),
                }
            )
        _emit_replay_log("fixed_scaffold_replay_heartbeat", **payload)
        last_progress_emit_s = float(elapsed_s)

    _emit_replay_log(
        "fixed_scaffold_replay_started",
        optimizer_method=("SPSA" if optimizer_method_key == "spsa" else "Powell"),
        optimizer_seed=int(optimizer_seed),
        optimizer_maxiter=int(optimizer_maxiter),
        optimizer_wallclock_cap_s=int(optimizer_wallclock_cap_s),
        shots=int(shots),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        transpile_optimization_level=int(transpile_optimization_level),
        seed_transpiler=(None if seed_transpiler is None else int(seed_transpiler)),
        runtime_parameter_count=int(theta0.size),
        term_order_id=str(subject.term_order_id),
    )
    _emit_progress(
        "fixed_scaffold_noisy_replay_initialized",
        objective_calls_total=0,
        best_call_index=None,
        runtime_job_ids_total=0,
        elapsed_s=float(time.perf_counter() - t0),
        partial_payload=_partial_payload(
            reason="in_progress",
            optimizer_stop_reason="in_progress",
            optimizer_message="in_progress",
        ),
    )

    with ExpectationOracle(noisy_cfg) as noisy_oracle:
        def _objective(theta: np.ndarray) -> float:
            nonlocal objective_calls_total, best_eval, wallclock_hit
            elapsed_s = float(time.perf_counter() - t0)
            if elapsed_s >= float(optimizer_wallclock_cap_s) and objective_calls_total > 0:
                wallclock_hit = True
                raise _WallclockStop("fixed scaffold noisy replay wallclock cap reached")
            theta_arr = np.asarray(theta, dtype=float).reshape(-1)
            qc = _build_ansatz_circuit(layout, theta_arr, int(nq), ref_state=ref_state)
            next_call_index = int(objective_calls_total + 1)
            job_events: list[dict[str, Any]] = []
            est = noisy_oracle.evaluate(
                qc,
                qop,
                runtime_job_observer=job_events.append,
                runtime_trace_context={
                    "call_index": int(next_call_index),
                    "phase": "optimizer",
                    "route": "fixed_scaffold_noisy_replay",
                },
            )
            objective_calls_total += 1
            job_records_by_id: dict[str, dict[str, Any]] = {}
            for evt in job_events:
                if not isinstance(evt, Mapping):
                    continue
                job = evt.get("job", {})
                if not isinstance(job, Mapping):
                    continue
                job_id = str(job.get("job_id", "")).strip()
                if job_id:
                    job_records_by_id[job_id] = dict(job)
            runtime_jobs = list(job_records_by_id.values())
            for rec in runtime_jobs:
                job_id = str(rec.get("job_id", "")).strip()
                if job_id and job_id not in runtime_job_ids:
                    runtime_job_ids.append(job_id)
            row = {
                "call_index": int(objective_calls_total),
                "status": "completed",
                "elapsed_s": float(time.perf_counter() - t0),
                "energy_noisy_mean": float(est.mean),
                "energy_noisy_stderr": float(est.stderr),
                "theta_runtime": [float(x) for x in theta_arr.tolist()],
                "theta_logical": [
                    float(x)
                    for x in np.asarray(
                        project_runtime_theta_block_mean(theta_arr, layout), dtype=float
                    ).tolist()
                ],
                "runtime_jobs": runtime_jobs,
            }
            _bounded_append(objective_history_tail, row)
            _bounded_append(objective_trace, row, limit=2000)
            if (
                best_eval is None
                or float(est.mean) < float(best_eval["energy_noisy_mean"])
                or (
                    float(est.mean) == float(best_eval["energy_noisy_mean"])
                    and float(est.stderr) < float(best_eval["energy_noisy_stderr"])
                )
            ):
                best_eval = {
                    "energy_noisy_mean": float(est.mean),
                    "energy_noisy_stderr": float(est.stderr),
                    "theta_runtime": np.asarray(theta_arr, dtype=float),
                    "call_index": int(objective_calls_total),
                }
                _emit_replay_log(
                    "fixed_scaffold_replay_best_update",
                    call_index=int(objective_calls_total),
                    elapsed_s=float(row["elapsed_s"]),
                    best_energy_noisy_mean=float(est.mean),
                    best_energy_noisy_stderr=float(est.stderr),
                )
            _emit_progress(
                "fixed_scaffold_noisy_replay_objective_completed",
                call_index=int(objective_calls_total),
                objective_calls_total=int(objective_calls_total),
                current_energy_noisy_mean=float(est.mean),
                best_call_index=(
                    None if best_eval is None else int(best_eval["call_index"])
                ),
                runtime_job_ids_total=int(len(runtime_job_ids)),
                elapsed_s=float(row["elapsed_s"]),
                partial_payload=_partial_payload(
                    reason="in_progress",
                    current_backend_info=_current_backend_info_payload(noisy_oracle),
                    optimizer_stop_reason="in_progress",
                    optimizer_message="in_progress",
                ),
            )
            _maybe_emit_progress(
                call_index=int(objective_calls_total),
                elapsed_s=float(row["elapsed_s"]),
                current_energy=float(est.mean),
            )
            return float(est.mean)

        try:
            if optimizer_method_key == "spsa":
                def _spsa_callback(payload: dict[str, Any]) -> None:
                    nonlocal last_spsa_payload
                    last_spsa_payload = dict(payload)
                    _maybe_emit_progress(
                        call_index=int(objective_calls_total),
                        elapsed_s=float(time.perf_counter() - t0),
                    )

                res = spsa_minimize(
                    fun=_objective,
                    x0=theta0,
                    maxiter=int(optimizer_maxiter),
                    seed=int(optimizer_seed),
                    a=float(spsa_a),
                    c=float(spsa_c),
                    alpha=float(spsa_alpha),
                    gamma=float(spsa_gamma),
                    A=float(spsa_A),
                    bounds=None,
                    project="none",
                    eval_repeats=int(spsa_eval_repeats),
                    eval_agg=str(spsa_eval_agg),
                    avg_last=int(spsa_avg_last),
                    callback=_spsa_callback,
                    callback_every=1,
                )
            else:
                try:
                    from scipy.optimize import minimize as scipy_minimize  # type: ignore
                except Exception as exc:
                    raise RuntimeError("SciPy minimize is unavailable for Powell fixed scaffold noisy replay.") from exc
                res = scipy_minimize(
                    _objective,
                    theta0,
                    method="Powell",
                    options={"maxiter": int(optimizer_maxiter)},
                )
        except _WallclockStop:
            wallclock_hit = True
        except Exception as exc:
            optimizer_exception = exc

        backend_info = {
            "noise_mode": str(noisy_oracle.backend_info.noise_mode),
            "estimator_kind": str(noisy_oracle.backend_info.estimator_kind),
            "backend_name": noisy_oracle.backend_info.backend_name,
            "using_fake_backend": bool(noisy_oracle.backend_info.using_fake_backend),
            "details": dict(noisy_oracle.backend_info.details),
        }

    if optimizer_exception is not None:
        optimizer_error = f"{type(optimizer_exception).__name__}: {optimizer_exception}"
        if objective_trace or best_eval is not None or runtime_job_ids:
            failure_payload = _partial_payload(
                reason="optimizer_exception",
                current_backend_info=backend_info,
                optimizer_stop_reason="optimizer_exception",
                optimizer_message=str(optimizer_error),
            )
            failure_payload["error"] = str(optimizer_error)
            return failure_payload
        return {
            "success": False,
            "available": False,
            "reason": "optimizer_exception",
            "error": str(optimizer_error),
            "artifact_json": str(ctx["path"]),
            "structure_locked": True,
            "pool_type": str(subject.pool_type),
            **_locked_subject_payload_fields(subject),
        }
    if wallclock_hit and best_eval is None:
        return {
            "success": False,
            "available": False,
            "reason": "wallclock_cap_before_first_eval",
            "artifact_json": str(ctx["path"]),
            "structure_locked": True,
            "pool_type": str(subject.pool_type),
            **_locked_subject_payload_fields(subject),
        }

    theta_best = (
        np.asarray(best_eval["theta_runtime"], dtype=float)
        if wallclock_hit and best_eval is not None
        else np.asarray(res.x if res is not None else theta0, dtype=float).reshape(-1)
    )
    stop_reason = "wallclock_cap" if wallclock_hit else "optimizer_complete"
    optimizer_success = bool(wallclock_hit or (res is not None and bool(res.success)))
    optimizer_message = "best-so-far returned at wallclock cap"
    optimizer_nfev = int(objective_calls_total)
    optimizer_nit = 0 if res is None else int(res.nit)
    if res is not None:
        optimizer_message = str(res.message)
        optimizer_nfev = int(res.nfev)

    initial_circuit = _build_ansatz_circuit(layout, theta0, int(nq), ref_state=ref_state)
    best_circuit = _build_ansatz_circuit(layout, theta_best, int(nq), ref_state=ref_state)
    initial_eval = _evaluate_locked_imported_circuit_energy(
        circuit=initial_circuit,
        ordered_labels_exyz=ordered_labels_exyz,
        static_coeff_map_exyz=static_coeff_map_exyz,
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        mitigation_config=mitigation_cfg,
        symmetry_mitigation_config=symmetry_cfg,
        backend_name=backend_name,
        use_fake_backend=bool(use_fake_backend),
        allow_aer_fallback=bool(allow_aer_fallback),
        omp_shm_workaround=bool(omp_shm_workaround),
        seed_transpiler=(None if seed_transpiler is None else int(seed_transpiler)),
        transpile_optimization_level=int(transpile_optimization_level),
        runtime_profile_config=runtime_profile_cfg,
        runtime_session_config=runtime_session_cfg,
    )
    best_runtime_energy_audits: dict[str, Any] | None = None
    if str(noisy_mode) == "runtime":
        best_runtime_energy_audits = _run_locked_imported_runtime_phase_evals(
            circuit=best_circuit,
            ordered_labels_exyz=ordered_labels_exyz,
            static_coeff_map_exyz=static_coeff_map_exyz,
            shots=int(shots),
            seed=int(seed),
            oracle_repeats=int(oracle_repeats),
            oracle_aggregate=str(oracle_aggregate),
            mitigation_config=mitigation_cfg,
            symmetry_mitigation_config=symmetry_cfg,
            backend_name=backend_name,
            allow_aer_fallback=bool(allow_aer_fallback),
            omp_shm_workaround=bool(omp_shm_workaround),
            seed_transpiler=(None if seed_transpiler is None else int(seed_transpiler)),
            transpile_optimization_level=int(transpile_optimization_level),
            runtime_session_config=runtime_session_cfg,
            main_runtime_profile_config=runtime_profile_cfg,
            dd_probe_runtime_profile_config=(
                normalize_runtime_estimator_profile_config("dd_probe_twirled_readout_v1")
                if bool(include_dd_probe)
                else None
            ),
            final_audit_runtime_profile_config=(
                normalize_runtime_estimator_profile_config("final_audit_zne_twirled_readout_v1")
                if bool(include_final_zne_audit)
                else None
            ),
        )
        best_main = (
            best_runtime_energy_audits.get("main", {})
            if isinstance(best_runtime_energy_audits, Mapping)
            else {}
        )
        best_eval_final = (
            dict(best_main.get("evaluation", {})) if isinstance(best_main, Mapping) else {}
        )
    else:
        best_eval_final = _evaluate_locked_imported_circuit_energy(
            circuit=best_circuit,
            ordered_labels_exyz=ordered_labels_exyz,
            static_coeff_map_exyz=static_coeff_map_exyz,
            shots=int(shots),
            seed=int(seed),
            oracle_repeats=int(oracle_repeats),
            oracle_aggregate=str(oracle_aggregate),
            mitigation_config=mitigation_cfg,
            symmetry_mitigation_config=symmetry_cfg,
            backend_name=backend_name,
            use_fake_backend=bool(use_fake_backend),
            allow_aer_fallback=bool(allow_aer_fallback),
            omp_shm_workaround=bool(omp_shm_workaround),
            seed_transpiler=(None if seed_transpiler is None else int(seed_transpiler)),
            transpile_optimization_level=int(transpile_optimization_level),
            runtime_profile_config=runtime_profile_cfg,
            runtime_session_config=runtime_session_cfg,
        )

    saved_theta_local_mitigation_ablation = _run_saved_theta_local_mitigation_ablation(
        circuit=initial_circuit,
        artifact_json=str(ctx["path"]),
        subject=subject,
        ordered_labels_exyz=ordered_labels_exyz,
        static_coeff_map_exyz=static_coeff_map_exyz,
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        backend_name=backend_name,
        allow_aer_fallback=bool(allow_aer_fallback),
        omp_shm_workaround=bool(omp_shm_workaround),
        seed_transpiler=(None if seed_transpiler is None else int(seed_transpiler)),
        transpile_optimization_level=int(transpile_optimization_level),
        local_dd_probe_sequence=local_dd_probe_sequence,
    )

    theta_best_logical = np.asarray(project_runtime_theta_block_mean(theta_best, layout), dtype=float)
    best_so_far_payload = _best_so_far_payload()
    _emit_replay_log(
        "fixed_scaffold_replay_completed",
        stop_reason=str(stop_reason),
        objective_calls_total=int(objective_calls_total),
        best_call_index=(None if best_eval is None else int(best_eval["call_index"])),
        best_noisy_minus_ideal=float(best_eval_final["delta_mean"]),
        elapsed_s=float(time.perf_counter() - t0),
    )
    return {
        "success": bool(optimizer_success),
        "available": True,
        "route": "fixed_scaffold_noisy_replay",
        "artifact_json": str(ctx["path"]),
        "source_kind": str(ctx.get("source_kind", "unknown")),
        "pool_type": str(subject.pool_type),
        "structure_locked": True,
        "matched_family_replay": False,
        "full_circuit_import_audit": False,
        "reps": 1,
        "partial": False,
        "theta_source": "imported_theta_runtime",
        "execution_mode": "backend_scheduled",
        "local_mitigation_label": str(local_mitigation_label),
        "reference_state_embedded": True,
        "ansatz_input_state_source": ansatz_input_state_meta.get("source", None),
        "ansatz_input_state_kind": ansatz_input_state_meta.get("handoff_state_kind", None),
        "parameterization": {
            "mode": "per_pauli_term_v1",
            "logical_parameter_count": int(layout.logical_parameter_count),
            "runtime_parameter_count": int(layout.runtime_parameter_count),
        },
        "optimizer": {
            "method": ("SPSA" if optimizer_method_key == "spsa" else "Powell"),
            "seed": int(optimizer_seed),
            "maxiter": int(optimizer_maxiter),
            "wallclock_cap_s": int(optimizer_wallclock_cap_s),
            "stop_reason": str(stop_reason),
            "iterations_completed": int(optimizer_nit),
            "objective_calls_total": int(optimizer_nfev),
            "message": str(optimizer_message),
        },
        "noise_config": {
            "noise_mode": str(noisy_mode),
            "shots": int(shots),
            "oracle_repeats": int(oracle_repeats),
            "oracle_aggregate": str(oracle_aggregate),
            "mitigation": dict(mitigation_cfg),
            "symmetry_mitigation": dict(symmetry_cfg),
            "runtime_profile": dict(runtime_profile_cfg),
            "runtime_session": dict(runtime_session_cfg),
        },
        "compile_control": {
            "transpile_optimization_level": int(transpile_optimization_level),
            "seed_transpiler": (None if seed_transpiler is None else int(seed_transpiler)),
            "source": "fake_backend_cli",
        },
        "theta": {
            "initial_runtime": [float(x) for x in theta0.tolist()],
            "best_runtime": [float(x) for x in theta_best.tolist()],
            "initial_logical": [float(x) for x in theta0_logical.tolist()],
            "best_logical": [float(x) for x in theta_best_logical.tolist()],
        },
        "energies": {
            "saved_artifact_energy": ctx.get("saved_energy", None),
            "initial_noisy_mean": float(initial_eval["noisy_mean"]),
            "initial_noisy_stderr": float(initial_eval["noisy_stderr"]),
            "initial_ideal_mean": float(initial_eval["ideal_mean"]),
            "initial_ideal_stderr": float(initial_eval["ideal_stderr"]),
            "best_noisy_mean": float(best_eval_final["noisy_mean"]),
            "best_noisy_stderr": float(best_eval_final["noisy_stderr"]),
            "best_ideal_mean": float(best_eval_final["ideal_mean"]),
            "best_ideal_stderr": float(best_eval_final["ideal_stderr"]),
            "best_noisy_minus_ideal": float(best_eval_final["delta_mean"]),
            "best_noisy_minus_ideal_stderr": float(best_eval_final["delta_stderr"]),
            "best_ideal_minus_saved_artifact": (
                None
                if ctx.get("saved_energy", None) is None
                else float(best_eval_final["ideal_mean"]) - float(ctx["saved_energy"])
            ),
        },
        "objective_history_tail": list(objective_history_tail),
        "objective_trace": list(objective_trace),
        "runtime_job_ids": [str(x) for x in runtime_job_ids],
        "best_so_far": dict(best_so_far_payload),
        "best_runtime_energy_audits": (
            dict(best_runtime_energy_audits) if isinstance(best_runtime_energy_audits, Mapping) else {}
        ),
        "backend_info": dict(best_eval_final.get("backend_info", backend_info)),
        "saved_theta_local_mitigation_ablation": dict(saved_theta_local_mitigation_ablation),
        "elapsed_s": float(time.perf_counter() - t0),
        **_locked_subject_payload_fields(subject),
    }


def _run_imported_fixed_scaffold_noise_attribution(
    *,
    artifact_json: str,
    shots: int,
    seed: int,
    oracle_repeats: int,
    oracle_aggregate: str,
    backend_name: str | None,
    use_fake_backend: bool,
    allow_aer_fallback: bool,
    omp_shm_workaround: bool,
    mitigation_config: dict[str, Any],
    symmetry_mitigation_config: dict[str, Any],
    slices: Sequence[str] = BACKEND_SCHEDULED_ATTRIBUTION_SLICES,
) -> dict[str, Any]:
    ctx, ansatz_input_state_meta, subject, error_payload = _resolve_locked_imported_fixed_scaffold_context(
        artifact_json,
        nonfixed_reason="fixed_scaffold_attribution_requires_locked_fixed_scaffold_source",
    )
    if error_payload is not None or ctx is None or subject is None:
        return dict(error_payload or {})

    mitigation_cfg = dict(normalize_mitigation_config(mitigation_config))
    symmetry_cfg = dict(normalize_symmetry_mitigation_config(symmetry_mitigation_config))
    if str(mitigation_cfg.get("mode", "none")) != "none":
        return {
            "success": False,
            "available": False,
            "reason": "fixed_scaffold_attribution_requires_mitigation_none",
            "artifact_json": str(ctx["path"]),
        }
    if str(symmetry_cfg.get("mode", "off")) != "off":
        return {
            "success": False,
            "available": False,
            "reason": "fixed_scaffold_attribution_requires_symmetry_off",
            "artifact_json": str(ctx["path"]),
        }

    requested_slices = tuple(str(x).strip().lower() for x in slices)
    qop = build_time_dependent_sparse_qop(
        ordered_labels_exyz=list(ctx["ordered_labels_exyz"]),
        static_coeff_map_exyz=dict(ctx["static_coeff_map_exyz"]),
        drive_coeff_map_exyz=None,
    )
    noisy_mode = "backend_scheduled"
    noisy_cfg = OracleConfig(
        noise_mode=str(noisy_mode),
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        backend_name=(None if backend_name is None else str(backend_name)),
        use_fake_backend=bool(use_fake_backend),
        allow_aer_fallback=bool(allow_aer_fallback),
        aer_fallback_mode="sampler_shots",
        omp_shm_workaround=bool(omp_shm_workaround),
        mitigation={"mode": "none", "zne_scales": [], "dd_sequence": None, "local_readout_strategy": None},
        symmetry_mitigation={"mode": "off"},
    )
    ideal_cfg = OracleConfig(
        noise_mode="ideal",
        shots=int(shots),
        seed=int(seed),
        oracle_repeats=int(oracle_repeats),
        oracle_aggregate=str(oracle_aggregate),
        backend_name=None,
        use_fake_backend=False,
        mitigation={"mode": "none", "zne_scales": [], "dd_sequence": None, "local_readout_strategy": None},
        symmetry_mitigation={"mode": "off"},
    )

    t0 = time.perf_counter()
    with ExpectationOracle(noisy_cfg) as noisy_oracle, ExpectationOracle(ideal_cfg) as ideal_oracle:
        ideal_est = ideal_oracle.evaluate(ctx["circuit"], qop)
        attribution = noisy_oracle.evaluate_backend_scheduled_attribution(
            ctx["circuit"],
            qop,
            slices=requested_slices,
        )

    ideal_reference = {
        "mean": float(ideal_est.mean),
        "std": float(ideal_est.std),
        "stdev": float(ideal_est.stdev),
        "stderr": float(ideal_est.stderr),
        "n_samples": int(ideal_est.n_samples),
        "raw_values": [float(x) for x in ideal_est.raw_values],
        "aggregate": str(ideal_est.aggregate),
    }
    slice_payloads: dict[str, Any] = {}
    successful_slices: list[str] = []
    for slice_name in requested_slices:
        rec = attribution.get("slices", {}).get(str(slice_name), {})
        if not isinstance(rec, Mapping):
            slice_payloads[str(slice_name)] = {
                "success": False,
                "slice": str(slice_name),
                "components": {},
                "noisy_mean": None,
                "noisy_std": None,
                "noisy_stdev": None,
                "noisy_stderr": None,
                "ideal_mean": float(ideal_est.mean),
                "ideal_std": float(ideal_est.std),
                "ideal_stdev": float(ideal_est.stdev),
                "ideal_stderr": float(ideal_est.stderr),
                "delta_mean": None,
                "delta_stderr": None,
                "backend_info": None,
                "reason": "missing_slice_payload",
                "error": None,
            }
            continue
        est = rec.get("estimate", None)
        success = bool(rec.get("success", False) and est is not None)
        if success:
            successful_slices.append(str(slice_name))
        noisy_mean = None if est is None else float(est.mean)
        noisy_std = None if est is None else float(est.std)
        noisy_stdev = None if est is None else float(est.stdev)
        noisy_stderr = None if est is None else float(est.stderr)
        delta_mean = None if est is None else float(est.mean - ideal_est.mean)
        delta_stderr = None if est is None else float(_combine_stderr(est.stderr, ideal_est.stderr))
        slice_payloads[str(slice_name)] = {
            "success": bool(success),
            "slice": str(slice_name),
            "components": dict(rec.get("components", {})) if isinstance(rec.get("components", {}), Mapping) else {},
            "noisy_mean": noisy_mean,
            "noisy_std": noisy_std,
            "noisy_stdev": noisy_stdev,
            "noisy_stderr": noisy_stderr,
            "ideal_mean": float(ideal_est.mean),
            "ideal_std": float(ideal_est.std),
            "ideal_stdev": float(ideal_est.stdev),
            "ideal_stderr": float(ideal_est.stderr),
            "delta_mean": delta_mean,
            "delta_stderr": delta_stderr,
            "backend_info": dict(rec.get("backend_info", {})) if isinstance(rec.get("backend_info", {}), Mapping) else None,
            "reason": rec.get("reason", None),
            "error": rec.get("error", None),
        }

    full_delta = slice_payloads.get("full", {}).get("delta_mean", None)
    readout_delta = slice_payloads.get("readout_only", {}).get("delta_mean", None)
    gate_delta = slice_payloads.get("gate_stateprep_only", {}).get("delta_mean", None)
    full_noisy = slice_payloads.get("full", {}).get("noisy_mean", None)
    readout_noisy = slice_payloads.get("readout_only", {}).get("noisy_mean", None)
    gate_noisy = slice_payloads.get("gate_stateprep_only", {}).get("noisy_mean", None)

    def _maybe_diff(a: Any, b: Any) -> float | None:
        if a is None or b is None:
            return None
        return float(a) - float(b)

    return {
        "success": bool(len(successful_slices) == len(requested_slices)),
        "available": True,
        "route": "fixed_scaffold_noise_attribution",
        "artifact_json": str(ctx["path"]),
        "source_kind": str(ctx.get("source_kind", "unknown")),
        "pool_type": str(subject.pool_type),
        "structure_locked": True,
        "matched_family_replay": False,
        "parameter_optimization": False,
        "reference_state_embedded": True,
        "ansatz_input_state_source": ansatz_input_state_meta.get("source", None),
        "ansatz_input_state_kind": ansatz_input_state_meta.get("handoff_state_kind", None),
        "parameterization": {
            "mode": "per_pauli_term_v1",
            "logical_parameter_count": int(ctx["layout"].logical_parameter_count),
            "runtime_parameter_count": int(ctx["layout"].runtime_parameter_count),
            "theta_source": "imported_artifact_runtime_theta",
        },
        "noise_config": {
            "noise_mode": str(noisy_mode),
            "shots": int(shots),
            "oracle_repeats": int(oracle_repeats),
            "oracle_aggregate": str(oracle_aggregate),
            "mitigation": dict(mitigation_cfg),
            "symmetry_mitigation": dict(symmetry_cfg),
        },
        "shared_compile": dict(attribution.get("shared_compile", {})),
        "ideal_reference": ideal_reference,
        "slices": slice_payloads,
        "slice_comparisons": {
            "full_minus_readout_only": _maybe_diff(full_noisy, readout_noisy),
            "full_minus_gate_stateprep_only": _maybe_diff(full_noisy, gate_noisy),
            "component_additivity_residual": (
                None
                if full_delta is None or readout_delta is None or gate_delta is None
                else float(full_delta) - float(readout_delta) - float(gate_delta)
            ),
        },
        "elapsed_s": float(time.perf_counter() - t0),
        **_locked_subject_payload_fields(subject),
    }


def _noisy_worker_entry(queue: Any, kwargs: dict[str, Any]) -> None:
    try:
        payload = _run_noisy_method_trajectory(**kwargs)
        queue.put({"ok": True, "payload": payload})
    except Exception as exc:  # pragma: no cover - subprocess fault path
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _run_noisy_mode_isolated(
    *,
    kwargs: dict[str, Any],
    timeout_s: int,
) -> dict[str, Any]:
    latest_progress: dict[str, Any] | None = None
    latest_partial_payload: dict[str, Any] | None = None
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_noisy_worker_entry, args=(queue, kwargs), daemon=False)
    proc.start()
    proc.join(timeout=float(timeout_s))

    if proc.is_alive():
        proc.terminate()
        proc.join(5.0)
        return {
            "success": False,
            "env_blocked": True,
            "reason": f"timeout_after_{int(timeout_s)}s",
            "exitcode": proc.exitcode,
        }

    if int(proc.exitcode or 0) != 0:
        nonzero_payload = (
            dict(latest_partial_payload)
            if isinstance(latest_partial_payload, Mapping)
            else {}
        )
        if nonzero_payload:
            nonzero_payload.update(
                {
                    "success": False,
                    "env_blocked": True,
                    "reason": "subprocess_nonzero_exit",
                    "exitcode": int(proc.exitcode or 0),
                }
            )
            if isinstance(latest_progress, Mapping):
                nonzero_payload["last_progress_event"] = latest_progress.get("event", None)
            return nonzero_payload
        return {
            "success": False,
            "env_blocked": True,
            "reason": "subprocess_nonzero_exit",
            "exitcode": int(proc.exitcode or 0),
        }

    if queue.empty():
        return {
            "success": False,
            "env_blocked": True,
            "reason": "subprocess_completed_without_payload",
            "exitcode": int(proc.exitcode or 0),
        }

    msg = queue.get()
    if not bool(msg.get("ok", False)):
        return {
            "success": False,
            "env_blocked": True,
            "reason": "worker_exception",
            "error": str(msg.get("error", "unknown")),
            "exitcode": int(proc.exitcode or 0),
        }

    return dict(msg.get("payload", {}))


def _noisy_audit_worker_entry(queue: Any, kwargs: dict[str, Any]) -> None:
    try:
        payload = _run_noisy_final_state_audit(**kwargs)
        queue.put({"ok": True, "payload": payload})
    except Exception as exc:  # pragma: no cover - subprocess fault path
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _imported_prepared_state_audit_worker_entry(queue: Any, kwargs: dict[str, Any]) -> None:
    try:
        payload = _run_imported_prepared_state_audit(**kwargs)
        queue.put({"ok": True, "payload": payload})
    except Exception as exc:  # pragma: no cover - subprocess fault path
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _imported_ansatz_input_state_audit_worker_entry(
    queue: Any,
    kwargs: dict[str, Any],
) -> None:
    try:
        payload = _run_imported_ansatz_input_state_audit(**kwargs)
        queue.put({"ok": True, "payload": payload})
    except Exception as exc:  # pragma: no cover - subprocess fault path
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _imported_full_circuit_audit_worker_entry(queue: Any, kwargs: dict[str, Any]) -> None:
    try:
        payload = _run_imported_full_circuit_audit(**kwargs)
        queue.put({"ok": True, "payload": payload})
    except Exception as exc:  # pragma: no cover - subprocess fault path
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _imported_fixed_lean_noisy_replay_worker_entry(queue: Any, kwargs: dict[str, Any]) -> None:
    try:
        payload = _run_imported_fixed_lean_noisy_replay(**kwargs)
        queue.put({"ok": True, "payload": payload})
    except Exception as exc:  # pragma: no cover - subprocess fault path
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _imported_fixed_lean_noise_attribution_worker_entry(queue: Any, kwargs: dict[str, Any]) -> None:
    try:
        payload = _run_imported_fixed_lean_noise_attribution(**kwargs)
        queue.put({"ok": True, "payload": payload})
    except Exception as exc:  # pragma: no cover - subprocess fault path
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _imported_fixed_lean_compile_control_scout_worker_entry(queue: Any, kwargs: dict[str, Any]) -> None:
    try:
        payload = _run_imported_fixed_lean_compile_control_scout(**kwargs)
        queue.put({"ok": True, "payload": payload})
    except Exception as exc:  # pragma: no cover - subprocess fault path
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _imported_fixed_scaffold_compile_control_scout_worker_entry(queue: Any, kwargs: dict[str, Any]) -> None:
    def _progress(payload: dict[str, Any]) -> None:
        queue.put({"kind": "progress", "payload": dict(payload)})

    try:
        payload = _run_imported_fixed_scaffold_compile_control_scout(
            progress_callback=_progress,
            **kwargs,
        )
        queue.put({"kind": "result", "ok": True, "payload": payload})
    except Exception as exc:  # pragma: no cover - subprocess fault path
        queue.put({"kind": "result", "ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _imported_fixed_scaffold_saved_theta_mitigation_matrix_worker_entry(
    queue: Any,
    kwargs: dict[str, Any],
) -> None:
    def _progress(payload: dict[str, Any]) -> None:
        queue.put({"kind": "progress", "payload": dict(payload)})

    try:
        payload = _run_imported_fixed_scaffold_saved_theta_mitigation_matrix(
            progress_callback=_progress,
            **kwargs,
        )
        queue.put({"kind": "result", "ok": True, "payload": payload})
    except Exception as exc:  # pragma: no cover - subprocess fault path
        queue.put({"kind": "result", "ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _imported_fixed_scaffold_runtime_energy_only_worker_entry(queue: Any, kwargs: dict[str, Any]) -> None:
    try:
        payload = _run_imported_fixed_scaffold_runtime_energy_only(**kwargs)
        queue.put({"ok": True, "payload": payload})
    except Exception as exc:  # pragma: no cover - subprocess fault path
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _imported_fixed_scaffold_runtime_raw_baseline_worker_entry(queue: Any, kwargs: dict[str, Any]) -> None:
    try:
        payload = _run_imported_fixed_scaffold_runtime_raw_baseline(**kwargs)
        queue.put({"ok": True, "payload": payload})
    except Exception as exc:  # pragma: no cover - subprocess fault path
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _imported_fixed_scaffold_noisy_replay_worker_entry(queue: Any, kwargs: dict[str, Any]) -> None:
    def _progress(payload: dict[str, Any]) -> None:
        queue.put({"kind": "progress", "payload": dict(payload)})

    try:
        payload = _run_imported_fixed_scaffold_noisy_replay(
            progress_callback=_progress,
            **kwargs,
        )
        queue.put({"kind": "result", "ok": True, "payload": payload})
    except Exception as exc:  # pragma: no cover - subprocess fault path
        queue.put({"kind": "result", "ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _imported_fixed_scaffold_noise_attribution_worker_entry(queue: Any, kwargs: dict[str, Any]) -> None:
    try:
        payload = _run_imported_fixed_scaffold_noise_attribution(**kwargs)
        queue.put({"ok": True, "payload": payload})
    except Exception as exc:  # pragma: no cover - subprocess fault path
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _run_noisy_audit_mode_isolated(
    *,
    kwargs: dict[str, Any],
    timeout_s: int,
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_noisy_audit_worker_entry, args=(queue, kwargs), daemon=False)
    proc.start()
    proc.join(timeout=float(timeout_s))

    if proc.is_alive():
        proc.terminate()
        proc.join(5.0)
        return {
            "success": False,
            "env_blocked": True,
            "reason": f"timeout_after_{int(timeout_s)}s",
            "exitcode": proc.exitcode,
        }

    if int(proc.exitcode or 0) != 0:
        return {
            "success": False,
            "env_blocked": True,
            "reason": "subprocess_nonzero_exit",
            "exitcode": int(proc.exitcode or 0),
        }

    if queue.empty():
        return {
            "success": False,
            "env_blocked": True,
            "reason": "subprocess_completed_without_payload",
            "exitcode": int(proc.exitcode or 0),
        }

    msg = queue.get()
    if not bool(msg.get("ok", False)):
        return {
            "success": False,
            "env_blocked": True,
            "reason": "worker_exception",
            "error": str(msg.get("error", "unknown")),
            "exitcode": int(proc.exitcode or 0),
        }
    return dict(msg.get("payload", {}))


def _run_imported_prepared_state_audit_mode_isolated(
    *,
    kwargs: dict[str, Any],
    timeout_s: int,
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_imported_prepared_state_audit_worker_entry, args=(queue, kwargs), daemon=False)
    proc.start()
    proc.join(timeout=float(timeout_s))
    if proc.is_alive():
        proc.terminate()
        proc.join(5.0)
        return {
            "success": False,
            "env_blocked": True,
            "reason": f"timeout_after_{int(timeout_s)}s",
            "exitcode": proc.exitcode,
        }
    if int(proc.exitcode or 0) != 0:
        return {
            "success": False,
            "env_blocked": True,
            "reason": "subprocess_nonzero_exit",
            "exitcode": int(proc.exitcode or 0),
        }
    if queue.empty():
        return {
            "success": False,
            "env_blocked": True,
            "reason": "subprocess_completed_without_payload",
            "exitcode": int(proc.exitcode or 0),
        }
    msg = queue.get()
    if not bool(msg.get("ok", False)):
        return {
            "success": False,
            "env_blocked": True,
            "reason": "worker_exception",
            "error": str(msg.get("error", "unknown")),
            "exitcode": int(proc.exitcode or 0),
        }
    return dict(msg.get("payload", {}))


def _run_imported_ansatz_input_state_audit_mode_isolated(
    *,
    kwargs: dict[str, Any],
    timeout_s: int,
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_imported_ansatz_input_state_audit_worker_entry,
        args=(queue, kwargs),
        daemon=False,
    )
    proc.start()
    proc.join(timeout=float(timeout_s))
    if proc.is_alive():
        proc.terminate()
        proc.join(5.0)
        return {
            "success": False,
            "env_blocked": True,
            "reason": f"timeout_after_{int(timeout_s)}s",
            "exitcode": proc.exitcode,
        }
    if int(proc.exitcode or 0) != 0:
        return {
            "success": False,
            "env_blocked": True,
            "reason": "subprocess_nonzero_exit",
            "exitcode": int(proc.exitcode or 0),
        }
    if queue.empty():
        return {
            "success": False,
            "env_blocked": True,
            "reason": "subprocess_completed_without_payload",
            "exitcode": int(proc.exitcode or 0),
        }
    msg = queue.get()
    if not bool(msg.get("ok", False)):
        return {
            "success": False,
            "env_blocked": True,
            "reason": "worker_exception",
            "error": str(msg.get("error", "unknown")),
            "exitcode": int(proc.exitcode or 0),
        }
    return dict(msg.get("payload", {}))


def _run_imported_full_circuit_audit_mode_isolated(
    *,
    kwargs: dict[str, Any],
    timeout_s: int,
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_imported_full_circuit_audit_worker_entry, args=(queue, kwargs), daemon=False)
    proc.start()
    proc.join(timeout=float(timeout_s))
    if proc.is_alive():
        proc.terminate()
        proc.join(5.0)
        return {
            "success": False,
            "env_blocked": True,
            "reason": f"timeout_after_{int(timeout_s)}s",
            "exitcode": proc.exitcode,
        }
    if int(proc.exitcode or 0) != 0:
        return {
            "success": False,
            "env_blocked": True,
            "reason": "subprocess_nonzero_exit",
            "exitcode": int(proc.exitcode or 0),
        }
    if queue.empty():
        return {
            "success": False,
            "env_blocked": True,
            "reason": "subprocess_completed_without_payload",
            "exitcode": int(proc.exitcode or 0),
        }
    msg = queue.get()
    if not bool(msg.get("ok", False)):
        return {
            "success": False,
            "env_blocked": True,
            "reason": "worker_exception",
            "error": str(msg.get("error", "unknown")),
            "exitcode": int(proc.exitcode or 0),
        }
    return dict(msg.get("payload", {}))


def _run_imported_fixed_lean_noisy_replay_mode_isolated(
    *,
    kwargs: dict[str, Any],
    timeout_s: int,
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_imported_fixed_lean_noisy_replay_worker_entry, args=(queue, kwargs), daemon=False)
    proc.start()
    proc.join(timeout=float(timeout_s))
    if proc.is_alive():
        proc.terminate()
        proc.join(5.0)
        return {
            "success": False,
            "env_blocked": True,
            "reason": f"timeout_after_{int(timeout_s)}s",
            "exitcode": proc.exitcode,
        }
    if int(proc.exitcode or 0) != 0:
        return {
            "success": False,
            "env_blocked": True,
            "reason": "subprocess_nonzero_exit",
            "exitcode": int(proc.exitcode or 0),
        }
    if queue.empty():
        return {
            "success": False,
            "env_blocked": True,
            "reason": "subprocess_completed_without_payload",
            "exitcode": int(proc.exitcode or 0),
        }
    msg = queue.get()
    if not bool(msg.get("ok", False)):
        return {
            "success": False,
            "env_blocked": True,
            "reason": "worker_exception",
            "error": str(msg.get("error", "unknown")),
            "exitcode": int(proc.exitcode or 0),
        }
    return dict(msg.get("payload", {}))


def _run_imported_fixed_lean_noise_attribution_mode_isolated(
    *,
    kwargs: dict[str, Any],
    timeout_s: int,
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_imported_fixed_lean_noise_attribution_worker_entry,
        args=(queue, kwargs),
        daemon=False,
    )
    proc.start()
    proc.join(timeout=float(timeout_s))
    if proc.is_alive():
        proc.terminate()
        proc.join(5.0)
        return {
            "success": False,
            "env_blocked": True,
            "reason": f"timeout_after_{int(timeout_s)}s",
            "exitcode": proc.exitcode,
        }
    if int(proc.exitcode or 0) != 0:
        return {
            "success": False,
            "env_blocked": True,
            "reason": "subprocess_nonzero_exit",
            "exitcode": int(proc.exitcode or 0),
        }
    if queue.empty():
        return {
            "success": False,
            "env_blocked": True,
            "reason": "subprocess_completed_without_payload",
            "exitcode": int(proc.exitcode or 0),
        }
    msg = queue.get()
    if not bool(msg.get("ok", False)):
        return {
            "success": False,
            "env_blocked": True,
            "reason": "worker_exception",
            "error": str(msg.get("error", "unknown")),
            "exitcode": int(proc.exitcode or 0),
        }
    return dict(msg.get("payload", {}))


def _run_imported_fixed_lean_compile_control_scout_mode_isolated(
    *,
    kwargs: dict[str, Any],
    timeout_s: int,
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_imported_fixed_lean_compile_control_scout_worker_entry,
        args=(queue, kwargs),
        daemon=False,
    )
    proc.start()
    proc.join(timeout=float(timeout_s))
    if proc.is_alive():
        proc.terminate()
        proc.join(5.0)
        return {
            "success": False,
            "env_blocked": True,
            "reason": f"timeout_after_{int(timeout_s)}s",
            "exitcode": proc.exitcode,
        }
    if int(proc.exitcode or 0) != 0:
        return {
            "success": False,
            "env_blocked": True,
            "reason": "subprocess_nonzero_exit",
            "exitcode": int(proc.exitcode or 0),
        }
    if queue.empty():
        return {
            "success": False,
            "env_blocked": True,
            "reason": "subprocess_completed_without_payload",
            "exitcode": int(proc.exitcode or 0),
        }
    msg = queue.get()
    if not bool(msg.get("ok", False)):
        return {
            "success": False,
            "env_blocked": True,
            "reason": "worker_exception",
            "error": str(msg.get("error", "unknown")),
            "exitcode": int(proc.exitcode or 0),
        }
    return dict(msg.get("payload", {}))


def _run_imported_fixed_scaffold_compile_control_scout_mode_isolated(
    *,
    kwargs: dict[str, Any],
    timeout_s: int,
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_imported_fixed_scaffold_compile_control_scout_worker_entry,
        args=(queue, kwargs),
        daemon=False,
    )
    proc.start()

    def _drain_messages() -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        result_msg: dict[str, Any] | None = None
        latest_progress: dict[str, Any] | None = None
        while True:
            try:
                msg = queue.get_nowait()
            except pyqueue.Empty:
                break
            kind = str(msg.get("kind", "result"))
            if kind == "progress":
                payload = msg.get("payload", {})
                if isinstance(payload, Mapping):
                    latest_progress = dict(payload)
            else:
                result_msg = dict(msg)
        return result_msg, latest_progress

    started_at = time.monotonic()
    deadline = started_at + float(timeout_s)
    result_msg: dict[str, Any] | None = None
    latest_progress: dict[str, Any] | None = None
    latest_partial_payload: dict[str, Any] | None = None

    def _finalize_failure_payload(
        *,
        reason: str,
        exitcode: int | None,
    ) -> dict[str, Any]:
        had_partial_payload = latest_partial_payload is not None
        payload = (
            dict(latest_partial_payload)
            if latest_partial_payload is not None
            else {}
        )
        payload.update(
            {
                "success": False,
                "env_blocked": True,
                "reason": str(reason),
                "exitcode": exitcode,
                "elapsed_s": float(max(0.0, time.monotonic() - started_at)),
            }
        )
        if isinstance(latest_progress, Mapping):
            payload["last_progress_event"] = latest_progress.get("event", None)
        return payload

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            break
        proc.join(timeout=min(0.25, remaining))
        drained_result, drained_progress = _drain_messages()
        if drained_progress is not None:
            latest_progress = dict(drained_progress)
            partial_payload = drained_progress.get("partial_payload", {})
            if partial_payload is not None:
                latest_partial_payload = dict(partial_payload)
        if drained_result is not None:
            result_msg = dict(drained_result)
        if result_msg is not None and not proc.is_alive():
            break
        if not proc.is_alive():
            break

    drained_result, drained_progress = _drain_messages()
    if drained_progress is not None:
        latest_progress = dict(drained_progress)
        partial_payload = drained_progress.get("partial_payload", {})
        if partial_payload is not None:
            latest_partial_payload = dict(partial_payload)
    if drained_result is not None:
        result_msg = dict(drained_result)

    if result_msg is not None:
        if proc.is_alive():
            proc.join(0.25)
            if proc.is_alive():
                proc.terminate()
                proc.join(5.0)
        if not bool(result_msg.get("ok", False)):
            worker_payload = _finalize_failure_payload(
                reason="worker_exception",
                exitcode=int(proc.exitcode or 0),
            )
            worker_payload["error"] = str(result_msg.get("error", "unknown"))
            return worker_payload
        return dict(result_msg.get("payload", {}))

    if proc.is_alive():
        proc.terminate()
        proc.join(5.0)
        timeout_payload = _finalize_failure_payload(
            reason=f"timeout_after_{int(timeout_s)}s",
            exitcode=proc.exitcode,
        )
        if timeout_payload:
            return timeout_payload
        return {
            "success": False,
            "env_blocked": True,
            "reason": f"timeout_after_{int(timeout_s)}s",
            "exitcode": proc.exitcode,
        }
    if int(proc.exitcode or 0) != 0:
        nonzero_payload = _finalize_failure_payload(
            reason="subprocess_nonzero_exit",
            exitcode=int(proc.exitcode or 0),
        )
        if nonzero_payload:
            return nonzero_payload
        return {
            "success": False,
            "env_blocked": True,
            "reason": "subprocess_nonzero_exit",
            "exitcode": int(proc.exitcode or 0),
        }
    if result_msg is None:
        grace_deadline = time.monotonic() + 0.25
        while time.monotonic() < grace_deadline and result_msg is None:
            drained_result, drained_progress = _drain_messages()
            if drained_progress is not None:
                latest_progress = dict(drained_progress)
                partial_payload = drained_progress.get("partial_payload", {})
                if partial_payload is not None:
                    latest_partial_payload = dict(partial_payload)
            if drained_result is not None:
                result_msg = dict(drained_result)
                break
            time.sleep(0.01)
    if result_msg is None:
        grace_deadline = time.monotonic() + 0.25
        while time.monotonic() < grace_deadline and result_msg is None:
            drained_result, drained_progress = _drain_messages()
            if drained_progress is not None:
                latest_progress = dict(drained_progress)
                partial_payload = drained_progress.get("partial_payload", {})
                if partial_payload is not None:
                    latest_partial_payload = dict(partial_payload)
            if drained_result is not None:
                result_msg = dict(drained_result)
                break
            time.sleep(0.01)
    if result_msg is None:
        grace_deadline = time.monotonic() + 0.25
        while time.monotonic() < grace_deadline and result_msg is None:
            drained_result, drained_progress = _drain_messages()
            if drained_progress is not None:
                latest_progress = dict(drained_progress)
                partial_payload = drained_progress.get("partial_payload", {})
                if partial_payload is not None:
                    latest_partial_payload = dict(partial_payload)
            if drained_result is not None:
                result_msg = dict(drained_result)
                break
            time.sleep(0.01)
    if result_msg is None:
        missing_payload = _finalize_failure_payload(
            reason="subprocess_completed_without_payload",
            exitcode=int(proc.exitcode or 0),
        )
        if missing_payload:
            return missing_payload
        return {
            "success": False,
            "env_blocked": True,
            "reason": "subprocess_completed_without_payload",
            "exitcode": int(proc.exitcode or 0),
        }
    msg = result_msg
    if not bool(msg.get("ok", False)):
        return {
            "success": False,
            "env_blocked": True,
            "reason": "worker_exception",
            "error": str(msg.get("error", "unknown")),
            "exitcode": int(proc.exitcode or 0),
        }
    return dict(msg.get("payload", {}))


def _run_imported_fixed_scaffold_runtime_energy_only_mode_isolated(
    *,
    kwargs: dict[str, Any],
    timeout_s: int,
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_imported_fixed_scaffold_runtime_energy_only_worker_entry,
        args=(queue, kwargs),
        daemon=False,
    )
    proc.start()
    proc.join(timeout=float(timeout_s))
    if proc.is_alive():
        proc.terminate()
        proc.join(5.0)
        return {
            "success": False,
            "env_blocked": True,
            "reason": f"timeout_after_{int(timeout_s)}s",
            "exitcode": proc.exitcode,
        }
    if int(proc.exitcode or 0) != 0:
        return {
            "success": False,
            "env_blocked": True,
            "reason": "subprocess_nonzero_exit",
            "exitcode": int(proc.exitcode or 0),
        }
    if queue.empty():
        return {
            "success": False,
            "env_blocked": True,
            "reason": "subprocess_completed_without_payload",
            "exitcode": int(proc.exitcode or 0),
        }
    msg = queue.get()
    if not bool(msg.get("ok", False)):
        return {
            "success": False,
            "env_blocked": True,
            "reason": "worker_exception",
            "error": str(msg.get("error", "unknown")),
            "exitcode": int(proc.exitcode or 0),
        }
    return dict(msg.get("payload", {}))


def _run_imported_fixed_scaffold_saved_theta_mitigation_matrix_mode_isolated(
    *,
    kwargs: dict[str, Any],
    timeout_s: int,
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_imported_fixed_scaffold_saved_theta_mitigation_matrix_worker_entry,
        args=(queue, kwargs),
        daemon=False,
    )
    proc.start()

    def _drain_messages() -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        result_msg: dict[str, Any] | None = None
        latest_progress: dict[str, Any] | None = None
        while True:
            try:
                msg = queue.get_nowait()
            except pyqueue.Empty:
                break
            kind = str(msg.get("kind", "result"))
            if kind == "progress":
                payload = msg.get("payload", {})
                if isinstance(payload, Mapping):
                    latest_progress = dict(payload)
            else:
                result_msg = dict(msg)
        return result_msg, latest_progress

    started_at = time.monotonic()
    deadline = started_at + float(timeout_s)
    result_msg: dict[str, Any] | None = None
    latest_progress: dict[str, Any] | None = None
    latest_partial_payload: dict[str, Any] | None = None

    def _finalize_failure_payload(
        *,
        reason: str,
        exitcode: int | None,
    ) -> dict[str, Any]:
        had_partial_payload = latest_partial_payload is not None
        payload = (
            dict(latest_partial_payload)
            if latest_partial_payload is not None
            else {}
        )
        payload.update(
            {
                "success": False,
                "env_blocked": True,
                "reason": str(reason),
                "exitcode": exitcode,
                "elapsed_s": float(max(0.0, time.monotonic() - started_at)),
            }
        )
        if isinstance(latest_progress, Mapping):
            payload["last_progress_event"] = latest_progress.get("event", None)
        return payload

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            break
        proc.join(timeout=min(0.25, remaining))
        drained_result, drained_progress = _drain_messages()
        if drained_progress is not None:
            latest_progress = dict(drained_progress)
            partial_payload = drained_progress.get("partial_payload", {})
            if partial_payload is not None:
                latest_partial_payload = dict(partial_payload)
        if drained_result is not None:
            result_msg = dict(drained_result)
        if result_msg is not None and not proc.is_alive():
            break
        if not proc.is_alive():
            break

    drained_result, drained_progress = _drain_messages()
    if drained_progress is not None:
        latest_progress = dict(drained_progress)
        partial_payload = drained_progress.get("partial_payload", {})
        if partial_payload is not None:
            latest_partial_payload = dict(partial_payload)
    if drained_result is not None:
        result_msg = dict(drained_result)

    if proc.is_alive():
        proc.terminate()
        proc.join(5.0)
        timeout_payload = _finalize_failure_payload(
            reason=f"timeout_after_{int(timeout_s)}s",
            exitcode=proc.exitcode,
        )
        if timeout_payload:
            return timeout_payload
        return {
            "success": False,
            "env_blocked": True,
            "reason": f"timeout_after_{int(timeout_s)}s",
            "exitcode": proc.exitcode,
        }
    if int(proc.exitcode or 0) != 0:
        nonzero_payload = _finalize_failure_payload(
            reason="subprocess_nonzero_exit",
            exitcode=int(proc.exitcode or 0),
        )
        if nonzero_payload:
            return nonzero_payload
        return {
            "success": False,
            "env_blocked": True,
            "reason": "subprocess_nonzero_exit",
            "exitcode": int(proc.exitcode or 0),
        }
    if result_msg is None:
        grace_deadline = time.monotonic() + 0.25
        while time.monotonic() < grace_deadline and result_msg is None:
            drained_result, drained_progress = _drain_messages()
            if drained_progress is not None:
                latest_progress = dict(drained_progress)
                partial_payload = drained_progress.get("partial_payload", {})
                if partial_payload is not None:
                    latest_partial_payload = dict(partial_payload)
            if drained_result is not None:
                result_msg = dict(drained_result)
                break
            time.sleep(0.01)
    if result_msg is None:
        missing_payload = _finalize_failure_payload(
            reason="subprocess_completed_without_payload",
            exitcode=int(proc.exitcode or 0),
        )
        if missing_payload:
            return missing_payload
        return {
            "success": False,
            "env_blocked": True,
            "reason": "subprocess_completed_without_payload",
            "exitcode": int(proc.exitcode or 0),
        }
    if not bool(result_msg.get("ok", False)):
        worker_payload = _finalize_failure_payload(
            reason="worker_exception",
            exitcode=int(proc.exitcode or 0),
        )
        worker_payload["error"] = str(result_msg.get("error", "unknown"))
        return worker_payload
    return dict(result_msg.get("payload", {}))


def _run_imported_fixed_scaffold_runtime_raw_baseline_mode_isolated(
    *,
    kwargs: dict[str, Any],
    timeout_s: int,
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_imported_fixed_scaffold_runtime_raw_baseline_worker_entry,
        args=(queue, kwargs),
        daemon=False,
    )
    proc.start()
    proc.join(timeout=float(timeout_s))
    if proc.is_alive():
        proc.terminate()
        proc.join(5.0)
        return {
            "success": False,
            "env_blocked": True,
            "reason": f"timeout_after_{int(timeout_s)}s",
            "exitcode": proc.exitcode,
        }
    if int(proc.exitcode or 0) != 0:
        return {
            "success": False,
            "env_blocked": True,
            "reason": "subprocess_nonzero_exit",
            "exitcode": int(proc.exitcode or 0),
        }
    if queue.empty():
        return {
            "success": False,
            "env_blocked": True,
            "reason": "subprocess_completed_without_payload",
            "exitcode": int(proc.exitcode or 0),
        }
    msg = queue.get()
    if not bool(msg.get("ok", False)):
        return {
            "success": False,
            "env_blocked": True,
            "reason": "worker_exception",
            "error": str(msg.get("error", "unknown")),
            "exitcode": int(proc.exitcode or 0),
        }
    return dict(msg.get("payload", {}))


def _run_imported_fixed_scaffold_noisy_replay_mode_isolated(
    *,
    kwargs: dict[str, Any],
    timeout_s: int,
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_imported_fixed_scaffold_noisy_replay_worker_entry,
        args=(queue, kwargs),
        daemon=False,
    )
    proc.start()

    def _drain_messages() -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        result_msg: dict[str, Any] | None = None
        latest_progress: dict[str, Any] | None = None
        while True:
            try:
                msg = queue.get_nowait()
            except pyqueue.Empty:
                break
            kind = str(msg.get("kind", "result"))
            if kind == "progress":
                payload = msg.get("payload", {})
                if isinstance(payload, Mapping):
                    latest_progress = dict(payload)
            else:
                result_msg = dict(msg)
        return result_msg, latest_progress

    started_at = time.monotonic()
    deadline = started_at + float(timeout_s)
    result_msg: dict[str, Any] | None = None
    latest_progress: dict[str, Any] | None = None
    latest_partial_payload: dict[str, Any] | None = None

    def _finalize_failure_payload(
        *,
        reason: str,
        exitcode: int | None,
    ) -> dict[str, Any]:
        had_partial_payload = latest_partial_payload is not None
        payload = (
            dict(latest_partial_payload)
            if latest_partial_payload is not None
            else {}
        )
        if isinstance(payload.get("optimizer", {}), Mapping):
            optimizer = dict(payload.get("optimizer", {}))
            optimizer["stop_reason"] = str(reason)
            payload["optimizer"] = optimizer
        payload.update(
            {
                "success": False,
                "env_blocked": True,
                "reason": str(reason),
                "exitcode": exitcode,
                "elapsed_s": float(max(0.0, time.monotonic() - started_at)),
            }
        )
        if had_partial_payload:
            payload.setdefault("partial", True)
        if isinstance(latest_progress, Mapping):
            payload["last_progress_event"] = latest_progress.get("event", None)
        return payload

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            break
        proc.join(timeout=min(0.25, remaining))
        drained_result, drained_progress = _drain_messages()
        if drained_progress is not None:
            latest_progress = dict(drained_progress)
            partial_payload = drained_progress.get("partial_payload", {})
            if partial_payload is not None:
                latest_partial_payload = dict(partial_payload)
        if drained_result is not None:
            result_msg = dict(drained_result)
        if result_msg is not None and not proc.is_alive():
            break
        if not proc.is_alive():
            break

    drained_result, drained_progress = _drain_messages()
    if drained_progress is not None:
        latest_progress = dict(drained_progress)
        partial_payload = drained_progress.get("partial_payload", {})
        if partial_payload is not None:
            latest_partial_payload = dict(partial_payload)
    if drained_result is not None:
        result_msg = dict(drained_result)

    if result_msg is not None:
        if proc.is_alive():
            proc.join(0.25)
            if proc.is_alive():
                proc.terminate()
                proc.join(5.0)
        if not bool(result_msg.get("ok", False)):
            worker_payload = _finalize_failure_payload(
                reason="worker_exception",
                exitcode=int(proc.exitcode or 0),
            )
            worker_payload["error"] = str(result_msg.get("error", "unknown"))
            return worker_payload
        return dict(result_msg.get("payload", {}))

    if proc.is_alive():
        proc.terminate()
        proc.join(5.0)
        timeout_payload = _finalize_failure_payload(
            reason=f"timeout_after_{int(timeout_s)}s",
            exitcode=proc.exitcode,
        )
        if timeout_payload:
            return timeout_payload
        return {
            "success": False,
            "env_blocked": True,
            "reason": f"timeout_after_{int(timeout_s)}s",
            "exitcode": proc.exitcode,
        }
    if int(proc.exitcode or 0) != 0:
        nonzero_payload = _finalize_failure_payload(
            reason="subprocess_nonzero_exit",
            exitcode=int(proc.exitcode or 0),
        )
        if nonzero_payload:
            return nonzero_payload
        return {
            "success": False,
            "env_blocked": True,
            "reason": "subprocess_nonzero_exit",
            "exitcode": int(proc.exitcode or 0),
        }
    if result_msg is None:
        grace_deadline = time.monotonic() + 0.25
        while time.monotonic() < grace_deadline and result_msg is None:
            drained_result, drained_progress = _drain_messages()
            if drained_progress is not None:
                latest_progress = dict(drained_progress)
                partial_payload = drained_progress.get("partial_payload", {})
                if partial_payload is not None:
                    latest_partial_payload = dict(partial_payload)
            if drained_result is not None:
                result_msg = dict(drained_result)
                break
            time.sleep(0.01)
    if result_msg is None:
        missing_payload = _finalize_failure_payload(
            reason="subprocess_completed_without_payload",
            exitcode=int(proc.exitcode or 0),
        )
        if missing_payload:
            return missing_payload
        return {
            "success": False,
            "env_blocked": True,
            "reason": "subprocess_completed_without_payload",
            "exitcode": int(proc.exitcode or 0),
        }
    if not bool(result_msg.get("ok", False)):
        worker_payload = _finalize_failure_payload(
            reason="worker_exception",
            exitcode=int(proc.exitcode or 0),
        )
        worker_payload["error"] = str(result_msg.get("error", "unknown"))
        return worker_payload
    return dict(result_msg.get("payload", {}))


def _run_imported_fixed_scaffold_noise_attribution_mode_isolated(
    *,
    kwargs: dict[str, Any],
    timeout_s: int,
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_imported_fixed_scaffold_noise_attribution_worker_entry,
        args=(queue, kwargs),
        daemon=False,
    )
    proc.start()
    proc.join(timeout=float(timeout_s))
    if proc.is_alive():
        proc.terminate()
        proc.join(5.0)
        return {
            "success": False,
            "env_blocked": True,
            "reason": f"timeout_after_{int(timeout_s)}s",
            "exitcode": proc.exitcode,
        }
    if int(proc.exitcode or 0) != 0:
        return {
            "success": False,
            "env_blocked": True,
            "reason": "subprocess_nonzero_exit",
            "exitcode": int(proc.exitcode or 0),
        }
    if queue.empty():
        return {
            "success": False,
            "env_blocked": True,
            "reason": "subprocess_completed_without_payload",
            "exitcode": int(proc.exitcode or 0),
        }
    msg = queue.get()
    if not bool(msg.get("ok", False)):
        return {
            "success": False,
            "env_blocked": True,
            "reason": "worker_exception",
            "error": str(msg.get("error", "unknown")),
            "exitcode": int(proc.exitcode or 0),
        }
    return dict(msg.get("payload", {}))


def _extract_series(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def _combine_stderr(noisy_stderr: float, ideal_stderr: float) -> float:
    n = float(noisy_stderr)
    i = float(ideal_stderr)
    if not np.isfinite(i):
        i = 0.0
    if not np.isfinite(n):
        n = 0.0
    return float(np.sqrt(max(0.0, n * n + i * i)))


def _delta_uncertainty_metrics(
    delta: np.ndarray,
    delta_stderr: np.ndarray,
) -> dict[str, float]:
    delta_abs = np.abs(np.asarray(delta, dtype=float))
    stderr_arr = np.asarray(delta_stderr, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(stderr_arr > 0.0, delta_abs / stderr_arr, np.nan)
    finite_z = z[np.isfinite(z)]
    return {
        "max_abs_delta": float(np.max(delta_abs)) if delta_abs.size > 0 else float("nan"),
        "max_abs_delta_over_stderr": (
            float(np.max(finite_z)) if finite_z.size > 0 else float("nan")
        ),
        "mean_abs_delta_over_stderr": (
            float(np.mean(finite_z)) if finite_z.size > 0 else float("nan")
        ),
    }


def _trajectory_delta_uncertainty(trajectory: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    if not trajectory:
        return {}
    channels = [
        "energy_static",
        "energy_total",
        "n_up_site0",
        "n_dn_site0",
        "doublon",
        "staggered",
    ]
    out: dict[str, dict[str, float]] = {}
    for ch in channels:
        delta_key = f"{ch}_delta_noisy_minus_ideal"
        delta_stderr_key = f"{delta_key}_stderr"
        if delta_key not in trajectory[0]:
            continue
        delta = np.asarray([float(r[delta_key]) for r in trajectory], dtype=float)
        delta_stderr = np.asarray(
            [float(r.get(delta_stderr_key, float("nan"))) for r in trajectory], dtype=float
        )
        out[ch] = _delta_uncertainty_metrics(delta, delta_stderr)
    return out


def _compute_exact_reference_for_hh(
    *,
    hmat: np.ndarray,
    num_sites: int,
    ordering: str,
    num_particles: tuple[int, int],
    energy_tol: float = 0.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (E0, psi_exact_gs, basis_gs_manifold) for HH sector-filtered reference."""
    gs_energy, basis = hc_pipeline._ground_manifold_basis_sector_filtered_hh(
        hmat=np.asarray(hmat, dtype=complex),
        num_sites=int(num_sites),
        num_particles=(int(num_particles[0]), int(num_particles[1])),
        ordering=str(ordering),
        nq_total=int(round(math.log2(int(np.asarray(hmat).shape[0])))),
        energy_tol=float(energy_tol),
    )
    psi_exact = hc_pipeline._normalize_state(
        np.asarray(basis[:, 0], dtype=complex).reshape(-1)
    )
    basis_orth = hc_pipeline._orthonormalize_basis_columns(
        np.asarray(basis, dtype=complex)
    )
    return float(gs_energy), np.asarray(psi_exact, dtype=complex), np.asarray(basis_orth, dtype=complex)


def _run_hardcoded_suzuki_profile(
    *,
    args: argparse.Namespace,
    hmat: np.ndarray,
    ordered_labels_exyz: list[str],
    coeff_map_exyz: dict[str, complex],
    psi_warm: np.ndarray,
    psi_adapt: np.ndarray,
    psi_final: np.ndarray,
    psi_exact_ref: np.ndarray,
    fidelity_basis_v0: np.ndarray,
    fidelity_subspace_energy_tol: float,
    drive_profile: dict[str, Any] | None,
) -> dict[str, Any]:
    """Run hardcoded-style suzuki2 trajectory with three mapped branches."""
    nq = int(round(math.log2(int(np.asarray(psi_final).size))))
    drive_provider_exyz, drive_meta = _drive_provider_from_profile(
        profile=drive_profile,
        num_sites=int(args.L),
        nq_total=int(nq),
        ordering=str(args.ordering),
    )

    rows, _ = hc_pipeline._simulate_trajectory(
        num_sites=int(args.L),
        ordering=str(args.ordering),
        psi0_legacy_trot=np.asarray(psi_warm, dtype=complex),
        psi0_paop_trot=np.asarray(psi_adapt, dtype=complex),
        psi0_hva_trot=np.asarray(psi_final, dtype=complex),
        legacy_branch_label="warm_start",
        psi0_exact_ref=np.asarray(psi_exact_ref, dtype=complex),
        fidelity_subspace_basis_v0=np.asarray(fidelity_basis_v0, dtype=complex),
        fidelity_subspace_energy_tol=float(fidelity_subspace_energy_tol),
        hmat=np.asarray(hmat, dtype=complex),
        ordered_labels_exyz=list(ordered_labels_exyz),
        coeff_map_exyz=dict(coeff_map_exyz),
        trotter_steps=int(args.trotter_steps),
        t_final=float(args.t_final),
        num_times=int(args.num_times),
        suzuki_order=2,
        drive_coeff_provider_exyz=drive_provider_exyz,
        drive_t0=float(0.0 if drive_profile is None else drive_profile.get("t0", 0.0)),
        drive_time_sampling=str(
            "midpoint" if drive_profile is None else drive_profile.get("time_sampling", "midpoint")
        ),
        exact_steps_multiplier=(
            int(args.exact_steps_multiplier) if drive_profile is not None else 1
        ),
        propagator="suzuki2",
        cfqm_stage_exp=str(args.cfqm_stage_exp),
        cfqm_coeff_drop_abs_tol=float(args.cfqm_coeff_drop_abs_tol),
        cfqm_normalize=bool(args.cfqm_normalize),
    )

    return {
        "drive_enabled": bool(drive_profile is not None),
        "drive_profile": drive_profile,
        "drive_meta": drive_meta,
        "branch_semantics": {
            "legacy": "warm_start_hva",
            "paop": "adapt_pool_b",
            "hva": "final_seeded_conventional_vqe",
        },
        "trajectory": rows,
        "fidelity_subspace_energy_tol": float(fidelity_subspace_energy_tol),
        "final": {
            "fidelity_legacy": float(rows[-1]["fidelity"]),
            "fidelity_paop": float(rows[-1]["fidelity_paop_trotter"]),
            "fidelity_hva": float(rows[-1]["fidelity_hva_trotter"]),
            "energy_total_trotter_legacy": float(rows[-1]["energy_total_trotter"]),
            "energy_total_trotter_paop": float(rows[-1]["energy_total_trotter_paop"]),
            "energy_total_trotter_hva": float(rows[-1]["energy_total_trotter_hva"]),
        },
    }


def _run_noiseless_profile(
    *,
    args: argparse.Namespace,
    psi_seed: np.ndarray,
    hmat: np.ndarray,
    ordered_labels_exyz: list[str],
    coeff_map_exyz: dict[str, complex],
    drive_profile: dict[str, Any] | None,
) -> dict[str, Any]:
    nq = int(round(math.log2(int(np.asarray(psi_seed).size))))
    drive_provider_exyz, drive_meta = _drive_provider_from_profile(
        profile=drive_profile,
        num_sites=int(args.L),
        nq_total=int(nq),
        ordering=str(args.ordering),
    )

    methods = [
        ("suzuki2", "suzuki2"),
        ("magnus2", "piecewise_exact"),
        ("cfqm4", "cfqm4"),
        ("cfqm6", "cfqm6"),
    ]

    method_payloads: dict[str, Any] = {}
    reference_rows: list[dict[str, Any]] | None = None

    for method_name, propagator_key in methods:
        rows, _ = hc_pipeline._simulate_trajectory(
            num_sites=int(args.L),
            ordering=str(args.ordering),
            psi0_legacy_trot=np.asarray(psi_seed, dtype=complex),
            psi0_paop_trot=np.asarray(psi_seed, dtype=complex),
            psi0_hva_trot=np.asarray(psi_seed, dtype=complex),
            legacy_branch_label="seq",
            psi0_exact_ref=np.asarray(psi_seed, dtype=complex),
            fidelity_subspace_basis_v0=np.asarray(psi_seed, dtype=complex).reshape(-1, 1),
            fidelity_subspace_energy_tol=0.0,
            hmat=np.asarray(hmat, dtype=complex),
            ordered_labels_exyz=list(ordered_labels_exyz),
            coeff_map_exyz=dict(coeff_map_exyz),
            trotter_steps=int(args.trotter_steps),
            t_final=float(args.t_final),
            num_times=int(args.num_times),
            suzuki_order=2,
            drive_coeff_provider_exyz=drive_provider_exyz,
            drive_t0=float(0.0 if drive_profile is None else drive_profile.get("t0", 0.0)),
            drive_time_sampling=str(
                "midpoint" if drive_profile is None else drive_profile.get("time_sampling", "midpoint")
            ),
            exact_steps_multiplier=(
                int(args.exact_steps_multiplier) if drive_profile is not None else 1
            ),
            propagator=str(propagator_key),
            cfqm_stage_exp=str(args.cfqm_stage_exp),
            cfqm_coeff_drop_abs_tol=float(args.cfqm_coeff_drop_abs_tol),
            cfqm_normalize=bool(args.cfqm_normalize),
        )
        if reference_rows is None:
            reference_rows = rows

        method_payloads[method_name] = {
            "propagator": str(propagator_key),
            "trajectory": rows,
            "final": {
                "energy_total_trotter": float(rows[-1]["energy_total_trotter"]),
                "energy_total_exact": float(rows[-1]["energy_total_exact"]),
                "abs_energy_total_error": float(
                    abs(float(rows[-1]["energy_total_trotter"]) - float(rows[-1]["energy_total_exact"]))
                ),
                "fidelity": float(rows[-1]["fidelity"]),
                "doublon_trotter": float(rows[-1]["doublon_trotter"]),
                "doublon_exact": float(rows[-1]["doublon_exact"]),
            },
        }

    assert reference_rows is not None

    return {
        "drive_enabled": bool(drive_profile is not None),
        "drive_profile": drive_profile,
        "drive_meta": drive_meta,
        "times": [float(r["time"]) for r in reference_rows],
        "reference": {
            "energy_static_exact": [float(r["energy_static_exact"]) for r in reference_rows],
            "energy_total_exact": [float(r["energy_total_exact"]) for r in reference_rows],
            "n_up_site0_exact": [float(r["n_up_site0_exact"]) for r in reference_rows],
            "n_dn_site0_exact": [float(r["n_dn_site0_exact"]) for r in reference_rows],
            "doublon_exact": [float(r["doublon_exact"]) for r in reference_rows],
            "staggered_exact": [float(r["staggered_exact"]) for r in reference_rows],
            "method": (
                "eigendecomposition"
                if drive_profile is None
                else str(reference_method_name(str(drive_profile.get("time_sampling", "midpoint"))) )
            ),
        },
        "methods": method_payloads,
    }


def _compute_comparisons(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "noiseless_vs_exact": {},
        "noise_vs_noiseless": {},
        "noise_vs_noiseless_methods": {},
        "noise_final_audit": {},
    }

    noiseless = payload.get("dynamics_noiseless", {})
    for profile_name, profile_data in noiseless.get("profiles", {}).items():
        methods = profile_data.get("methods", {})
        profile_cmp: dict[str, Any] = {}
        for method_name, method_data in methods.items():
            final = method_data.get("final", {})
            profile_cmp[method_name] = {
                "final_abs_energy_total_error": float(final.get("abs_energy_total_error", float("nan"))),
                "final_fidelity": float(final.get("fidelity", float("nan"))),
            }
        out["noiseless_vs_exact"][str(profile_name)] = profile_cmp

    noisy = payload.get("dynamics_noisy", {})
    for profile_name, profile_data in noisy.get("profiles", {}).items():
        method_payloads = profile_data.get("methods", {})
        if not isinstance(method_payloads, dict) or not method_payloads:
            method_payloads = {"suzuki2": {"modes": profile_data.get("modes", {})}}

        method_cmp: dict[str, Any] = {}
        for method_name, method_data in method_payloads.items():
            modes = method_data.get("modes", {}) if isinstance(method_data, dict) else {}
            comp_profile: dict[str, Any] = {}

            noisl_ref = (
                payload.get("dynamics_noiseless", {})
                .get("profiles", {})
                .get(profile_name, {})
                .get("methods", {})
                .get(str(method_name), {})
                .get("trajectory", [])
            )
            if not noisl_ref:
                noisl_ref = (
                    payload.get("dynamics_noiseless", {})
                    .get("profiles", {})
                    .get(profile_name, {})
                    .get("methods", {})
                    .get("suzuki2", {})
                    .get("trajectory", [])
                )

            e_ref_final = None
            d_ref_final = None
            if noisl_ref:
                e_ref_final = float(noisl_ref[-1]["energy_total_trotter"])
                d_ref_final = float(noisl_ref[-1]["doublon_trotter"])

            for mode_name, mode_data in modes.items():
                if not bool(mode_data.get("success", False)):
                    comp_profile[str(mode_name)] = {
                        "available": False,
                        "reason": str(mode_data.get("reason", mode_data.get("error", "unknown"))),
                    }
                    continue
                traj = mode_data.get("trajectory", [])
                if not traj:
                    comp_profile[str(mode_name)] = {"available": False, "reason": "empty_trajectory"}
                    continue

                e_noisy_final = float(traj[-1]["energy_total_noisy"])
                d_noisy_final = float(traj[-1]["doublon_noisy"])
                rec = {
                    "available": True,
                    "final_energy_total_noisy": e_noisy_final,
                    "final_doublon_noisy": d_noisy_final,
                    "final_energy_total_delta_noisy_minus_ideal": float(
                        traj[-1]["energy_total_delta_noisy_minus_ideal"]
                    ),
                    "final_energy_total_delta_noisy_minus_ideal_stderr": float(
                        traj[-1].get("energy_total_delta_noisy_minus_ideal_stderr", float("nan"))
                    ),
                    "final_doublon_delta_noisy_minus_ideal": float(
                        traj[-1]["doublon_delta_noisy_minus_ideal"]
                    ),
                    "final_doublon_delta_noisy_minus_ideal_stderr": float(
                        traj[-1].get("doublon_delta_noisy_minus_ideal_stderr", float("nan"))
                    ),
                }
                delta_unc = mode_data.get("delta_uncertainty", {})
                if isinstance(delta_unc, dict):
                    rec["delta_uncertainty"] = dict(delta_unc)
                if e_ref_final is not None:
                    rec[f"final_energy_total_delta_noisy_minus_noiseless_{method_name}"] = float(e_noisy_final - e_ref_final)
                if d_ref_final is not None:
                    rec[f"final_doublon_delta_noisy_minus_noiseless_{method_name}"] = float(d_noisy_final - d_ref_final)
                comp_profile[str(mode_name)] = rec
            method_cmp[str(method_name)] = comp_profile

        out["noise_vs_noiseless_methods"][str(profile_name)] = method_cmp
        out["noise_vs_noiseless"][str(profile_name)] = dict(method_cmp.get("suzuki2", {}))

    noisy_audit = payload.get("noisy_final_audit", {})
    for profile_name, profile_data in noisy_audit.get("profiles", {}).items():
        modes = profile_data.get("modes", {}) if isinstance(profile_data, dict) else {}
        profile_cmp: dict[str, Any] = {}
        for mode_name, mode_data in modes.items():
            if not bool(isinstance(mode_data, dict) and mode_data.get("success", False)):
                profile_cmp[str(mode_name)] = {
                    "available": False,
                    "reason": str(mode_data.get("reason", mode_data.get("error", "unknown"))),
                }
                continue
            obs = mode_data.get("final_observables", {}) if isinstance(mode_data, dict) else {}
            e_total = obs.get("energy_total", {}) if isinstance(obs.get("energy_total", {}), dict) else {}
            doublon = obs.get("doublon", {}) if isinstance(obs.get("doublon", {}), dict) else {}
            rec = {
                "available": True,
                "final_energy_total_noisy": float(e_total.get("noisy_mean", float("nan"))),
                "final_doublon_noisy": float(doublon.get("noisy_mean", float("nan"))),
                "final_energy_total_delta_noisy_minus_ideal": float(e_total.get("delta_mean", float("nan"))),
                "final_energy_total_delta_noisy_minus_ideal_stderr": float(
                    e_total.get("delta_stderr", float("nan"))
                ),
                "final_doublon_delta_noisy_minus_ideal": float(doublon.get("delta_mean", float("nan"))),
                "final_doublon_delta_noisy_minus_ideal_stderr": float(
                    doublon.get("delta_stderr", float("nan"))
                ),
            }
            delta_unc = mode_data.get("delta_uncertainty", {})
            if isinstance(delta_unc, dict):
                rec["delta_uncertainty"] = dict(delta_unc)
            profile_cmp[str(mode_name)] = rec
        out["noise_final_audit"][str(profile_name)] = profile_cmp

    return out


def _build_summary(payload: dict[str, Any]) -> dict[str, Any]:
    stage_pipeline = payload.get("stage_pipeline", {})
    warm = stage_pipeline.get("warm_start", {})
    adapt = stage_pipeline.get("adapt_pool_b", {})
    final = stage_pipeline.get("conventional_vqe", {})

    noisy = payload.get("dynamics_noisy", {}).get("profiles", {})
    noisy_completed = 0
    noisy_total = 0
    noisy_method_modes_completed = 0
    noisy_method_modes_total = 0
    delta_abs_vals: list[float] = []
    delta_sig_vals: list[float] = []
    delta_sig_mean_vals: list[float] = []
    audit_delta_abs_vals: list[float] = []
    audit_delta_sig_vals: list[float] = []
    audit_delta_sig_mean_vals: list[float] = []
    for prof in noisy.values():
        modes = prof.get("modes", {}) if isinstance(prof, dict) else {}
        for mode_data in modes.values():
            noisy_total += 1
            if bool(mode_data.get("success", False)):
                noisy_completed += 1
        methods = prof.get("methods", {}) if isinstance(prof, dict) else {}
        if isinstance(methods, dict):
            for method_data in methods.values():
                method_modes = method_data.get("modes", {}) if isinstance(method_data, dict) else {}
                for mode_data in method_modes.values():
                    noisy_method_modes_total += 1
                    if bool(mode_data.get("success", False)):
                        noisy_method_modes_completed += 1
                    delta_unc = mode_data.get("delta_uncertainty", {}) if isinstance(mode_data, dict) else {}
                    if isinstance(delta_unc, dict):
                        for rec in delta_unc.values():
                            if not isinstance(rec, dict):
                                continue
                            max_abs = float(rec.get("max_abs_delta", float("nan")))
                            max_sig = float(rec.get("max_abs_delta_over_stderr", float("nan")))
                            mean_sig = float(rec.get("mean_abs_delta_over_stderr", float("nan")))
                            if np.isfinite(max_abs):
                                delta_abs_vals.append(max_abs)
                            if np.isfinite(max_sig):
                                delta_sig_vals.append(max_sig)
                            if np.isfinite(mean_sig):
                                delta_sig_mean_vals.append(mean_sig)

    noisy_audit = payload.get("noisy_final_audit", {}).get("profiles", {})
    noisy_audit_modes_completed = 0
    noisy_audit_modes_total = 0
    for prof in noisy_audit.values():
        modes = prof.get("modes", {}) if isinstance(prof, dict) else {}
        for mode_data in modes.values():
            noisy_audit_modes_total += 1
            if bool(isinstance(mode_data, dict) and mode_data.get("success", False)):
                noisy_audit_modes_completed += 1
            delta_unc = mode_data.get("delta_uncertainty", {}) if isinstance(mode_data, dict) else {}
            if isinstance(delta_unc, dict):
                for rec in delta_unc.values():
                    if not isinstance(rec, dict):
                        continue
                    max_abs = float(rec.get("max_abs_delta", float("nan")))
                    max_sig = float(rec.get("max_abs_delta_over_stderr", float("nan")))
                    mean_sig = float(rec.get("mean_abs_delta_over_stderr", float("nan")))
                    if np.isfinite(max_abs):
                        audit_delta_abs_vals.append(max_abs)
                    if np.isfinite(max_sig):
                        audit_delta_sig_vals.append(max_sig)
                    if np.isfinite(mean_sig):
                        audit_delta_sig_mean_vals.append(mean_sig)

    dyn_bench = payload.get("dynamics_benchmarks", {})
    dyn_rows = dyn_bench.get("rows", []) if isinstance(dyn_bench, dict) else []
    delta_abs_all = [*delta_abs_vals, *audit_delta_abs_vals]
    delta_sig_all = [*delta_sig_vals, *audit_delta_sig_vals]
    delta_sig_mean_all = [*delta_sig_mean_vals, *audit_delta_sig_mean_vals]

    return {
        "warm_delta_abs": float(warm.get("delta_abs", float("nan"))),
        "adapt_delta_abs": float(adapt.get("delta_abs", float("nan"))),
        "final_delta_abs": float(final.get("delta_abs", float("nan"))),
        "warm_stop_reason": str(warm.get("stop_reason", "")),
        "adapt_stop_reason": str(adapt.get("stop_reason", "")),
        "final_stop_reason": str(final.get("stop_reason", "")),
        "noisy_modes_completed": int(noisy_completed),
        "noisy_modes_total": int(noisy_total),
        "noisy_method_modes_completed": int(noisy_method_modes_completed),
        "noisy_method_modes_total": int(noisy_method_modes_total),
        "noisy_audit_modes_completed": int(noisy_audit_modes_completed),
        "noisy_audit_modes_total": int(noisy_audit_modes_total),
        "max_abs_delta": (
            float(np.max(np.asarray(delta_abs_all, dtype=float)))
            if delta_abs_all
            else float("nan")
        ),
        "max_abs_delta_over_stderr": (
            float(np.max(np.asarray(delta_sig_all, dtype=float)))
            if delta_sig_all
            else float("nan")
        ),
        "mean_abs_delta_over_stderr": (
            float(np.mean(np.asarray(delta_sig_mean_all, dtype=float)))
            if delta_sig_mean_all
            else float("nan")
        ),
        "noisy_audit_max_abs_delta": (
            float(np.max(np.asarray(audit_delta_abs_vals, dtype=float)))
            if audit_delta_abs_vals
            else float("nan")
        ),
        "noisy_audit_max_abs_delta_over_stderr": (
            float(np.max(np.asarray(audit_delta_sig_vals, dtype=float)))
            if audit_delta_sig_vals
            else float("nan")
        ),
        "noisy_audit_mean_abs_delta_over_stderr": (
            float(np.mean(np.asarray(audit_delta_sig_mean_vals, dtype=float)))
            if audit_delta_sig_mean_vals
            else float("nan")
        ),
        "dynamics_benchmark_rows": int(len(dyn_rows) if isinstance(dyn_rows, list) else 0),
    }


def _build_equation_registry_and_contracts(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build equation registry and plot contracts for equation-first audit pages."""
    registry: dict[str, Any] = {}

    def _add(
        eq_id: str,
        *,
        latex: str,
        plain: str,
        symbols: dict[str, str],
        units: str,
        source_keys: list[str],
    ) -> None:
        registry[str(eq_id)] = {
            "latex": str(latex),
            "plain": str(plain),
            "symbols": dict(symbols),
            "units": str(units),
            "source_keys": [str(x) for x in source_keys],
        }

    _add(
        "eq_h_total",
        latex=r"H(t)=H_0 + H_{\mathrm{drive}}(t)",
        plain="Total Hamiltonian splits into static + drive terms.",
        symbols={"H0": "static HH Hamiltonian", "H_drive": "time-dependent onsite-density drive"},
        units="energy",
        source_keys=["settings.t", "settings.u", "settings.g_ep", "settings.drive_profile"],
    )
    _add(
        "eq_h_drive_density",
        latex=r"H_{\mathrm{drive}}(t)=\sum_{i,\sigma} v_i(t)\,n_{i\sigma}",
        plain="Drive couples to onsite electron number operators.",
        symbols={"i": "site index", "sigma": "spin channel"},
        units="energy",
        source_keys=["settings.drive_profile", "diagnostics.metric_definitions.energy_total"],
    )
    _add(
        "eq_drive_waveform",
        latex=r"v_i(t)=s_i A\sin[\omega(t+t_0)+\phi]\exp\!\left(-\frac{(t+t_0)^2}{2\bar t^2}\right)",
        plain="Gaussian-envelope sinusoidal drive with spatial weights s_i.",
        symbols={"A": "drive amplitude", "omega": "carrier angular frequency", "tbar": "Gaussian width", "phi": "phase", "s_i": "site weight"},
        units="energy",
        source_keys=["settings.drive_profile.A", "settings.drive_profile.omega", "settings.drive_profile.tbar"],
    )
    _add(
        "eq_number_operator",
        latex=r"n_{i\sigma}=\frac{I-Z_{q(i,\sigma)}}{2}",
        plain="JW number-operator identity used for occupancy observables.",
        symbols={"q(i,sigma)": "qubit index for spin-orbital (i,sigma)"},
        units="dimensionless",
        source_keys=["diagnostics.metric_definitions"],
    )
    _add(
        "eq_total_density",
        latex=r"n_i=\langle n_{i\uparrow}+n_{i\downarrow}\rangle",
        plain="Site-resolved total occupancy.",
        symbols={"n_i": "site occupancy expectation"},
        units="particles",
        source_keys=["dynamics_noiseless.profiles.*.methods.suzuki2.trajectory"],
    )
    _add(
        "eq_doublon",
        latex=r"D_i=\langle n_{i\uparrow}n_{i\downarrow}\rangle",
        plain="Onsite doublon expectation.",
        symbols={"D_i": "double occupancy at site i"},
        units="dimensionless",
        source_keys=["dynamics_noiseless.profiles.*.methods.suzuki2.trajectory"],
    )
    _add(
        "eq_staggered",
        latex=r"S=\frac{1}{L}\sum_{i=0}^{L-1}(-1)^i\langle n_i\rangle",
        plain="Staggered charge-density order parameter.",
        symbols={"L": "number of sites"},
        units="dimensionless",
        source_keys=["dynamics_noiseless.profiles.*.methods.suzuki2.trajectory"],
    )
    _add(
        "eq_energy_static",
        latex=r"E_{\mathrm{static}}^b(t)=\langle\psi_b(t)|H_0|\psi_b(t)\rangle",
        plain="Static-Hamiltonian energy channel for branch b.",
        symbols={"b": "branch (warm/adapt/final)"},
        units="energy",
        source_keys=["dynamics_noiseless.profiles.*.methods.suzuki2.trajectory"],
    )
    _add(
        "eq_energy_total",
        latex=r"E_{\mathrm{total}}^b(t)=\langle\psi_b(t)|H_0+H_{\mathrm{drive}}(t)|\psi_b(t)\rangle",
        plain="Instantaneous total energy under drive.",
        symbols={"b": "branch (warm/adapt/final)"},
        units="energy",
        source_keys=["dynamics_noiseless.profiles.*.methods.suzuki2.trajectory"],
    )
    _add(
        "eq_subspace_fidelity",
        latex=r"F_{\mathrm{sub}}^b(t)=\langle\psi_b^{\mathrm{trot}}(t)|P_{\mathrm{GS}}(t)|\psi_b^{\mathrm{trot}}(t)\rangle",
        plain="Projected fidelity against sector-filtered exact GS manifold.",
        symbols={"P_GS": "projector onto exact GS manifold"},
        units="probability",
        source_keys=["dynamics_noiseless.profiles.*.methods.suzuki2.trajectory"],
    )
    _add(
        "eq_delta_e",
        latex=r"\Delta E_k=E_k-E_{\mathrm{exact,sector}},\quad \delta_k=|\Delta E_k|",
        plain="Stage transition error metrics.",
        symbols={"k": "checkpoint index"},
        units="energy",
        source_keys=["stage_pipeline.*.transition.delta_abs_trace"],
    )
    _add(
        "eq_slope_switch",
        latex=r"\mathrm{switch}\iff |\mathrm{slope}(\delta_{k-w+1:k})|\le \varepsilon\ \text{for}\ p\ \text{consecutive checkpoints}",
        plain="Windowed abs-slope plateau rule for stage transitions.",
        symbols={"w": "window_k", "epsilon": "slope threshold", "p": "patience"},
        units="energy/checkpoint",
        source_keys=["settings.transition_policy"],
    )
    _add(
        "eq_noisy_estimator",
        latex=(
            r"\mu_O=\mathrm{agg}(\{\langle O\rangle_r\}_{r=1}^{R}),\quad "
            r"\sigma_O=\mathrm{std}(\{\langle O\rangle_r\}_{r=1}^{R}),\quad "
            r"\mathrm{stderr}_O=\sigma_O/\sqrt{R}"
        ),
        plain="Noisy oracle aggregate, empirical spread, and standard error over repeats.",
        symbols={"R": "oracle repeats", "agg": "mean/median aggregate"},
        units="observable units",
        source_keys=["settings.oracle_repeats", "settings.oracle_aggregate"],
    )
    _add(
        "eq_noisy_delta",
        latex=(
            r"\Delta_O^{\mathrm{noise-ideal}}(t)=\mu_O^{\mathrm{noise}}(t)-\mu_O^{\mathrm{ideal}}(t),\quad "
            r"\mathrm{stderr}_{\Delta}=\sqrt{\mathrm{stderr}_{\mathrm{noise}}^2+\mathrm{stderr}_{\mathrm{ideal}}^2}"
        ),
        plain="Noisy-vs-ideal observable delta with propagated standard error.",
        symbols={"O": "observable channel"},
        units="observable units",
        source_keys=["dynamics_noisy.profiles.*.modes.*.trajectory.*.*_delta_noisy_minus_ideal"],
    )
    _add(
        "eq_proxy_cost",
        latex=r"\mathrm{cx\_proxy\_term}(p)=2\max(w(p)-1,0),\quad \mathrm{sq\_proxy\_term}(p)=2\,xy(p)+1",
        plain="Per-term hardware proxy definitions for dynamics benchmark totals.",
        symbols={"w(p)": "Pauli weight", "xy(p)": "count of X/Y letters"},
        units="proxy-count",
        source_keys=["dynamics_noisy.profiles.*.methods.*.modes.*.benchmark_cost"],
    )
    _add(
        "eq_runtime_bench",
        latex=r"t_{\mathrm{wall}}=t_{\mathrm{build}}+t_{\mathrm{oracle}}+\cdots",
        plain="Runtime decomposition for noisy dynamics evaluation.",
        symbols={"t_wall": "wall-clock total", "t_oracle": "oracle evaluation subtotal"},
        units="seconds",
        source_keys=["dynamics_noisy.profiles.*.methods.*.modes.*.benchmark_runtime"],
    )
    _add(
        "eq_suzuki2",
        latex=r"U_{\mathrm{S2}}(\Delta t)\approx\prod_j e^{-i\frac{\Delta t}{2}H_j}\prod_{j}^{\mathrm{rev}}e^{-i\frac{\Delta t}{2}H_j}",
        plain="Second-order symmetric Suzuki product step.",
        symbols={"H_j": "Hamiltonian term blocks"},
        units="unitary",
        source_keys=["dynamics_noiseless.profiles.*.methods.suzuki2"],
    )
    _add(
        "eq_magnus2",
        latex=r"U_{\mathrm{Magnus2}}(\Delta t)=\exp\!\left[-i\Delta t\,H\!\left(t+\frac{\Delta t}{2}\right)\right]",
        plain="Exponential midpoint (Magnus-2) reference approximation.",
        symbols={"Delta t": "macro step size"},
        units="unitary",
        source_keys=["dynamics_noiseless.profiles.*.methods.magnus2"],
    )
    _add(
        "eq_cfqm",
        latex=r"U_{\mathrm{CFQM}}(\Delta t)\approx\prod_{m} \exp[-i\,a_m\Delta t\,H(t+c_m\Delta t)]",
        plain="Commutator-free Magnus stage product (CFQM4/CFQM6).",
        symbols={"a_m,c_m": "scheme coefficients"},
        units="unitary",
        source_keys=["dynamics_noiseless.profiles.*.methods.cfqm4", "dynamics_noiseless.profiles.*.methods.cfqm6"],
    )
    _add(
        "eq_error_abs",
        latex=r"\epsilon_X(t)=|X_{\mathrm{approx}}(t)-X_{\mathrm{exact}}(t)|",
        plain="Absolute trajectory error metric.",
        symbols={"X": "observable channel"},
        units="observable units",
        source_keys=["comparisons.noiseless_vs_exact", "comparisons.noise_vs_noiseless"],
    )

    profiles = ["static", "drive"]
    observables = ["energy_total", "energy_static", "doublon", "staggered", "n_up_site0", "n_dn_site0"]
    for profile in profiles:
        for obs in observables:
            eq_id = f"eq_{obs}_{profile}_final_seed"
            _add(
                eq_id,
                latex=rf"{obs}^{{\mathrm{{final\_seed}}}}_{{{profile}}}(t)",
                plain=f"{obs} channel for {profile} profile, propagated from final-seed state only.",
                symbols={"profile": profile, "seed": "final VQE state"},
                units="observable units",
                source_keys=[f"dynamics_noiseless.profiles.{profile}.methods.suzuki2.trajectory"],
            )

    modes = ["ideal", "shots", "aer_noise"]
    noisy_obs = ["energy_total", "energy_static", "doublon", "staggered", "n_up_site0", "n_dn_site0"]
    for profile in profiles:
        for mode in modes:
            for obs in noisy_obs:
                eq_id = f"eq_noisy_{obs}_{profile}_{mode}"
                _add(
                    eq_id,
                    latex=rf"\mu_{{{obs}}}^{{{mode},{profile}}}(t)",
                    plain=f"Noisy estimate for {obs} in {profile}/{mode}.",
                    symbols={"mode": mode, "profile": profile},
                    units="observable units",
                    source_keys=[f"dynamics_noisy.profiles.{profile}.modes.{mode}.trajectory"],
                )

    contracts: dict[str, Any] = {}
    contracts["plot_stage_transition"] = {
        "x": "checkpoint index",
        "y": ["delta_abs_trace", "slope_trace"],
        "source": ["stage_pipeline.warm_start.transition", "stage_pipeline.adapt_pool_b.transition"],
        "notes": "Windowed abs-slope switching traces.",
    }
    for profile in profiles:
        contracts[f"plot_{profile}_magnus_cfqm_overlay"] = {
            "x": "time grid",
            "y": ["suzuki2.energy_total_trotter", "magnus2.energy_total_trotter", "cfqm4.energy_total_trotter", "cfqm6.energy_total_trotter", "exact.energy_total_exact"],
            "source": [f"dynamics_noiseless.profiles.{profile}.methods.*.trajectory"],
            "notes": "Noiseless method comparison.",
        }
        contracts[f"plot_{profile}_magnus_cfqm_error"] = {
            "x": "time grid",
            "y": ["|E_method-E_exact|", "|F_method-1|"],
            "source": [f"dynamics_noiseless.profiles.{profile}.methods.*.trajectory"],
            "notes": "Noiseless error channels vs exact.",
        }
        contracts[f"plot_{profile}_overlay_energy"] = {
            "x": "time grid",
            "y": ["noiseless+suzuki2+exact+noisy energy overlays"],
            "source": [f"dynamics_noisy.profiles.{profile}.modes.*.trajectory", f"dynamics_noiseless.profiles.{profile}.methods.suzuki2.trajectory"],
            "notes": "Overlay page for exact/noiseless/noisy energies.",
        }
        contracts[f"plot_{profile}_overlay_occupancy"] = {
            "x": "time grid",
            "y": ["n_up(0)", "n_dn(0) overlays"],
            "source": [f"dynamics_noisy.profiles.{profile}.modes.*.trajectory", f"dynamics_noiseless.profiles.{profile}.methods.suzuki2.trajectory"],
            "notes": "Overlay page for site-0 occupancies.",
        }
        contracts[f"plot_{profile}_overlay_doublon_staggered"] = {
            "x": "time grid",
            "y": ["doublon", "staggered overlays"],
            "source": [f"dynamics_noisy.profiles.{profile}.modes.*.trajectory", f"dynamics_noiseless.profiles.{profile}.methods.suzuki2.trajectory"],
            "notes": "Overlay page for doublon and staggered order.",
        }
        for mode in modes:
            contracts[f"plot_{profile}_{mode}_energy"] = {
                "x": "time grid",
                "y": ["energy_static_noisy", "energy_static_ideal", "energy_total_noisy", "energy_total_ideal"],
                "source": [f"dynamics_noisy.profiles.{profile}.modes.{mode}.trajectory"],
                "notes": "Noisy energy channels and noisy-ideal deltas.",
            }
            contracts[f"plot_{profile}_{mode}_occupancy"] = {
                "x": "time grid",
                "y": ["n_up_site0_noisy", "n_up_site0_ideal", "n_dn_site0_noisy", "n_dn_site0_ideal"],
                "source": [f"dynamics_noisy.profiles.{profile}.modes.{mode}.trajectory"],
                "notes": "Noisy site-0 spin channels and deltas.",
            }
            contracts[f"plot_{profile}_{mode}_doublon_staggered"] = {
                "x": "time grid",
                "y": ["doublon_noisy", "doublon_ideal", "staggered_noisy", "staggered_ideal"],
                "source": [f"dynamics_noisy.profiles.{profile}.modes.{mode}.trajectory"],
                "notes": "Noisy doublon/staggered channels and deltas.",
            }

    return registry, contracts


def _apply_report_theme(plt: Any) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.5,
            "figure.dpi": 220,
            "savefig.dpi": 220,
            "axes.grid": False,
        }
    )


def _render_page_header_footer(fig: Any, *, header: str, footer: str) -> None:
    fig.text(0.5, 0.985, str(header), ha="center", va="top", fontsize=9, color="#222222")
    fig.text(0.5, 0.012, str(footer), ha="center", va="bottom", fontsize=8, color="#333333")


_ACTIVE_CAPTION_OVERRIDES: dict[str, list[str]] = {}


def _build_caption_overrides(shots: int) -> dict[str, list[str]]:
    shots_text = f"{int(shots)} shots" if int(shots) > 0 else "configured shots"
    mode_shot_text = f"mode=ideal, {shots_text}"
    mapping: dict[str, list[str]] = {
        "plot_stage_transition": [
            "Stage switching: $|\\Delta E|$ vs checkpoint for warm start and ADAPT.",
            "Dashed trace is the windowed slope used by the plateau rule ($w=5$, $\\epsilon=5\\times10^{-5}$, patience=3).",
        ],
        "plot_static_density_total_heatmaps": [
            "Static: total occupancy n_i(t) heatmaps (sites i=0..L-1).",
            "Panels: exact GS (left), exact evolution (middle), noiseless ADAPT-HVA + Suzuki-2 (right).",
        ],
        "plot_static_density_up_heatmaps": [
            "Static: spin-up occupancy n_up_i(t) heatmaps.",
            "Panels: exact GS (left), exact evolution (middle), noiseless ADAPT-HVA + Suzuki-2 (right).",
        ],
        "plot_static_density_dn_heatmaps": [
            "Static: spin-down occupancy n_dn_i(t) heatmaps.",
            "Panels: exact GS (left), exact evolution (middle), noiseless ADAPT-HVA + Suzuki-2 (right).",
        ],
        "plot_static_energy_fidelity_final": [
            "Static (final-seed trajectory): top F_sub, middle E_total, bottom doublon vs time.",
            "Exact reference vs noiseless ADAPT-HVA + Suzuki-2.",
        ],
        "plot_drive_density_total_heatmaps": [
            "Drive: total occupancy n_i(t) heatmaps (sites i=0..L-1).",
            "Panels: exact GS (left), exact evolution (middle), noiseless ADAPT-HVA + Suzuki-2 (right).",
        ],
        "plot_drive_density_up_heatmaps": [
            "Drive: spin-up occupancy n_up_i(t) heatmaps.",
            "Panels: exact GS (left), exact evolution (middle), noiseless ADAPT-HVA + Suzuki-2 (right).",
        ],
        "plot_drive_density_dn_heatmaps": [
            "Drive: spin-down occupancy n_dn_i(t) heatmaps.",
            "Panels: exact GS (left), exact evolution (middle), noiseless ADAPT-HVA + Suzuki-2 (right).",
        ],
        "plot_drive_energy_fidelity_final": [
            "Drive (final-seed trajectory): top F_sub, middle E_total, bottom doublon vs time.",
            "Exact reference vs noiseless ADAPT-HVA + Suzuki-2.",
        ],
        "plot_static_magnus_cfqm_overlay": [
            "Static: noiseless integrator comparison - top E_total, bottom F_sub.",
            "Methods: Suzuki-2 / Magnus-2 / CFQM4 / CFQM6; exact reference in black.",
        ],
        "plot_static_magnus_cfqm_error": [
            "Static: integrator error vs exact - top |E-E_exact|, bottom (1-F_sub).",
            "Smaller is better for ranking Suzuki-2 vs Magnus/CFQM baselines.",
        ],
        "plot_drive_magnus_cfqm_overlay": [
            "Drive: noiseless integrator comparison - top E_total, bottom F_sub.",
            "Methods: Suzuki-2 / Magnus-2 / CFQM4 / CFQM6; exact reference in black.",
        ],
        "plot_drive_magnus_cfqm_error": [
            "Drive: integrator error vs exact - top |E-E_exact|, bottom (1-F_sub).",
            "Smaller is better for ranking Suzuki-2 vs Magnus/CFQM baselines.",
        ],
        "plot_static_overlay_energy": [
            "Static: energy vs time - top E_total, bottom E_static.",
            f"Exact vs noiseless (ADAPT-HVA+Suzuki-2) vs noisy ({mode_shot_text}).",
        ],
        "plot_static_overlay_occupancy": [
            "Static: site-0 occupations vs time - top $n_{\\uparrow}(0)$, bottom $n_{\\downarrow}(0)$.",
            f"Noiseless vs noisy ({mode_shot_text}).",
        ],
        "plot_static_overlay_doublon_staggered": [
            "Static: correlators vs time - top doublon, bottom staggered order S.",
            f"Noiseless vs noisy ({mode_shot_text}).",
        ],
        "plot_drive_overlay_energy": [
            "Drive: energy vs time - top E_total, bottom E_static.",
            f"Exact vs noiseless (ADAPT-HVA+Suzuki-2) vs noisy ({mode_shot_text}).",
        ],
        "plot_drive_overlay_occupancy": [
            "Drive: site-0 occupations vs time - top $n_{\\uparrow}(0)$, bottom $n_{\\downarrow}(0)$.",
            f"Noiseless vs noisy ({mode_shot_text}).",
        ],
        "plot_drive_overlay_doublon_staggered": [
            "Drive: correlators vs time - top doublon, bottom staggered order S.",
            f"Noiseless vs noisy ({mode_shot_text}).",
        ],
        "plot_static_scalar_error_heatmap": [
            "Static: absolute scalar error channels heatmap (rows are |E|, |D|, and |S| error metrics).",
            "Columns run over time; brighter color means larger deviation from the matched exact channel.",
        ],
        "plot_drive_scalar_error_heatmap": [
            "Drive: absolute scalar error channels heatmap (rows are |E|, |D|, and |S| error metrics).",
            "Columns run over time; brighter color means larger deviation from the matched exact channel.",
        ],
        "plot_drive_waveform": [
            "Drive waveform v(t) evaluated on the simulation time grid.",
            "Gaussian-envelope sinusoid used in H_drive(t)=sum_i,sigma v_i(t) n_i,sigma.",
        ],
        "plot_noisy_benchmark_table": [
            "Noisy dynamics benchmark table: profile/method/mode rows.",
            "Reports proxy costs (term-exp, cx, sq, depth) and runtime totals (wall/oracle).",
        ],
    }

    for profile in ("static", "drive"):
        profile_label = "Static" if profile == "static" else "Drive"
        for mode in ("ideal", "shots", "aer_noise"):
            mode_label = mode.replace("_", " ")
            mapping[f"plot_{profile}_{mode}_energy"] = [
                f"{profile_label} / mode={mode_label}: top E_total, middle E_static, bottom $\\Delta$(measured-ideal-ref).",
                "Shows estimator deviation from its internal ideal reference (not the ED exact curve).",
            ]
            mapping[f"plot_{profile}_{mode}_occupancy"] = [
                f"{profile_label} / mode={mode_label}: top $n_{{\\uparrow}}(0)$, middle $n_{{\\downarrow}}(0)$, bottom $\\Delta$(measured-ideal-ref).",
                "Use the $\\Delta$ panel to isolate estimator/noise bias over time.",
            ]
            mapping[f"plot_{profile}_{mode}_doublon_staggered"] = [
                f"{profile_label} / mode={mode_label}: top doublon, middle staggered S, bottom $\\Delta$(measured-ideal-ref).",
                "Use the $\\Delta$ panel to isolate estimator/noise bias over time.",
            ]
    return mapping


def _fallback_caption_line(plot_id: str, plot_contracts: dict[str, Any]) -> str:
    pid = str(plot_id)
    if pid.startswith("plot_static_"):
        section = "Static"
    elif pid.startswith("plot_drive_"):
        section = "Drive"
    elif pid.startswith("plot_stage_"):
        section = "Stage"
    else:
        section = "Report"
    rec = plot_contracts.get(pid, {})
    notes = str(rec.get("notes", "")).strip()
    if notes:
        title = notes.rstrip(".")
    else:
        title = pid.replace("plot_", "").replace("_", " ").strip()
    return f"{section}: {title}."


def _extract_numeric_caption_line(style_legend_lines: list[str]) -> str:
    for raw in style_legend_lines:
        line = str(raw).strip()
        if ("max|Δ|" in line) or ("RMS|Δ|" in line) or ("max|Delta|" in line) or ("RMS|Delta|" in line):
            return line.rstrip(".") + "."
    return ""


def _disabled_hardcoded_superset_meta() -> dict[str, Any]:
    return {
        "profiles": {},
        "disabled": True,
        "reason": "branch propagation deactivated; final-only dynamics",
    }


def _noise_style_legend_lines() -> list[str]:
    return [
        "noise ideal   : #ff7f0e",
        "noise shots   : #2ca02c",
        "noise aer     : #d62728",
        "noiseless (final-seed Suzuki-2) : #1f77b4",
        "exact reference                    : #111111",
        "solid mode color = noisy measured; dashed mode color = ideal reference for mode",
        "dotted black horizontal = zero baseline for Δ(noisy-ideal)",
        "translucent mode band = ±2*stderr around Δ(noisy-ideal)",
    ]


def _noise_config_caption(settings: dict[str, Any], mode: str) -> str:
    mitigation = settings.get("mitigation_config", {}) if isinstance(settings, dict) else {}
    symmetry = settings.get("symmetry_mitigation_config", {}) if isinstance(settings, dict) else {}
    return (
        "noise_config: "
        f"mode={str(mode)}, "
        f"shots={settings.get('shots')}, "
        f"oracle_repeats={settings.get('oracle_repeats')}, "
        f"oracle_aggregate={settings.get('oracle_aggregate')}, "
        f"mitigation={mitigation.get('mode', 'none')}, "
        f"zne_scales={mitigation.get('zne_scales', [])}, "
        f"dd_sequence={mitigation.get('dd_sequence', None)}, "
        f"symmetry={symmetry.get('mode', 'off')}"
    )


def _latex_to_unicode_math(expr: str) -> str:
    text = str(expr)
    if not text:
        return ""
    # Reduce common wrappers first.
    text = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    # Flatten simple fractions repeatedly.
    frac_pat = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    while True:
        new_text = frac_pat.sub(r"(\1)/(\2)", text)
        if new_text == text:
            break
        text = new_text
    repl = {
        r"\Delta": "Δ",
        r"\delta": "δ",
        r"\epsilon": "ε",
        r"\omega": "ω",
        r"\phi": "φ",
        r"\bar t": "t̄",
        r"\sum": "Σ",
        r"\prod": "∏",
        r"\approx": "≈",
        r"\le": "≤",
        r"\ge": "≥",
        r"\iff": "⇔",
        r"\cdot": "·",
        r"\times": "×",
        r"\langle": "⟨",
        r"\rangle": "⟩",
        r"\left": "",
        r"\right": "",
    }
    for src, dst in repl.items():
        text = text.replace(src, dst)
    text = text.replace("{", "").replace("}", "")
    text = text.replace("\\", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _latex_to_mathtext(expr: str) -> str:
    text = str(expr)
    if not text:
        return ""
    text = re.sub(r"\\text\{([^{}]*)\}", r"\\mathrm{\1}", text)
    text = text.replace(r"\iff", r"\Leftrightarrow")
    text = re.sub(r"\\le(?![A-Za-z])", r"\\leq", text)
    text = re.sub(r"\\ge(?![A-Za-z])", r"\\geq", text)
    text = text.replace(r"\quad", r"\;")
    text = text.replace(r"\!", "")
    return text


def _annotate_plot_with_equations(
    fig: Any,
    *,
    eq_ids: list[str],
    equation_registry: dict[str, Any],
    plot_id: str,
    plot_contracts: dict[str, Any],
    style_legend_lines: list[str],
) -> None:
    """Attach short human-readable caption strip below plots."""
    _ = eq_ids
    _ = equation_registry
    override = _ACTIVE_CAPTION_OVERRIDES.get(str(plot_id), [])
    if override:
        caption_lines = [str(x).strip() for x in override if str(x).strip()]
    else:
        caption_lines = [_fallback_caption_line(str(plot_id), plot_contracts)]

    numeric_line = _extract_numeric_caption_line(style_legend_lines)
    if numeric_line and len(caption_lines) < 3:
        caption_lines.append(numeric_line)

    caption_lines = caption_lines[:3]
    if caption_lines:
        caption_lines[0] = f"[PLOT_CAPTION] {caption_lines[0]}"
    else:
        caption_lines = ["[PLOT_CAPTION]"]

    caption = "\n".join(caption_lines[:3])
    fig.subplots_adjust(bottom=0.20, top=0.93)
    fig.text(
        0.02,
        0.04,
        caption,
        ha="left",
        va="bottom",
        fontsize=7.7,
        family="DejaVu Sans",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f7f7f7", "edgecolor": "#303030", "alpha": 0.98},
    )
    _render_page_header_footer(
        fig,
        header=f"Hubbard-Holstein report | plot={plot_id}",
        footer=f"generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}",
    )


def _matrix_from_rows(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([[float(v) for v in row[key]] for row in rows], dtype=float)


def _render_formula_atlas(
    pdf: Any,
    *,
    equation_registry: dict[str, Any],
    title: str = "SECTION: APPENDIX FORMULA ATLAS",
    keys: list[str] | None = None,
    include_sources: bool = True,
) -> None:
    require_matplotlib()
    plt = get_plt()
    selected = sorted(list(keys if keys is not None else equation_registry.keys()))
    if not selected:
        return
    per_page = 6 if include_sources else 8
    for start in range(0, len(selected), per_page):
        chunk_keys = selected[start : start + per_page]
        fig = plt.figure(figsize=(8.5, 11.0))
        fig.patch.set_facecolor("white")
        fig.text(0.06, 0.965, title, ha="left", va="top", fontsize=11, fontweight="bold")
        fig.text(0.06, 0.942, "Formula index for this report.", ha="left", va="top", fontsize=9)

        y = 0.91
        for eq_id in chunk_keys:
            rec = equation_registry.get(eq_id, {})
            fig.text(0.06, y, f"[{eq_id}]", ha="left", va="top", fontsize=8.5, fontweight="bold")
            y -= 0.020

            latex_expr = _latex_to_mathtext(rec.get("latex", ""))
            rendered = False
            if latex_expr:
                try:
                    fig.text(0.08, y, f"${latex_expr}$", ha="left", va="top", fontsize=10)
                    rendered = True
                except Exception:
                    rendered = False
            if not rendered:
                fig.text(0.08, y, _latex_to_unicode_math(rec.get("latex", "")), ha="left", va="top", fontsize=9)
            y -= 0.026

            plain = str(rec.get("plain", "")).strip()
            units = str(rec.get("units", "")).strip()
            if plain:
                fig.text(0.08, y, f"Meaning: {plain}", ha="left", va="top", fontsize=8)
                y -= 0.018
            if units:
                fig.text(0.08, y, f"Units: {units}", ha="left", va="top", fontsize=8)
                y -= 0.018

            if include_sources:
                src = rec.get("source_keys", [])
                src_list = [str(x) for x in src] if isinstance(src, list) else [str(src)]
                if len(src_list) > 2:
                    src_text = ", ".join(src_list[:2]) + f", ... (+{len(src_list)-2} more)"
                else:
                    src_text = ", ".join(src_list)
                fig.text(0.08, y, f"Source: {src_text}", ha="left", va="top", fontsize=7)
                y -= 0.018
            y -= 0.010

        _render_page_header_footer(
            fig,
            header="Hubbard-Holstein report | formula atlas",
            footer=f"generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}",
        )
        pdf.savefig(fig)
        plt.close(fig)


def _write_pdf(pdf_path: Path, payload: dict[str, Any]) -> None:
    require_matplotlib()
    plt = get_plt()
    PdfPages = get_PdfPages()
    _apply_report_theme(plt)

    settings = payload.get("settings", {})
    stage_pipeline = payload.get("stage_pipeline", {})
    pool_B = payload.get("pool_B", {})
    noiseless = payload.get("dynamics_noiseless", {}).get("profiles", {})
    noisy = payload.get("dynamics_noisy", {}).get("profiles", {})
    hardcoded_superset = payload.get("hardcoded_superset", {})
    comparisons = payload.get("comparisons", {})
    diagnostics = payload.get("diagnostics", {})
    equation_registry = payload.get("equation_registry", {})
    plot_contracts = payload.get("plot_contracts", {})
    global _ACTIVE_CAPTION_OVERRIDES
    _ACTIVE_CAPTION_OVERRIDES = _build_caption_overrides(int(settings.get("shots", 0)))

    if not isinstance(equation_registry, dict) or len(equation_registry) == 0:
        equation_registry, plot_contracts = _build_equation_registry_and_contracts(payload)

    branch_style_lines = [
        "color warm (legacy)  : #1f77b4",
        "color adapt (paop)   : #2ca02c",
        "color final (hva)    : #d62728",
        "color exact reference: #111111",
        "Warm/ADAPT/final are optimization checkpoints only.",
        "Dynamics/noise pages are seeded from final VQE only.",
    ]
    method_style_lines = [
        "method suzuki2: #1f77b4",
        "method magnus2: #ff7f0e",
        "method cfqm4  : #2ca02c",
        "method cfqm6  : #d62728",
        "exact ref     : #111111",
    ]
    noise_style_lines = _noise_style_legend_lines()
    summary = payload.get("summary", {})
    used_eq_ids: set[str] = set()
    disable_time_dynamics = bool(settings.get("disable_time_dynamics", False))
    drive_cfg = settings.get("drive_profile", {})
    drive_enabled = bool(isinstance(drive_cfg, dict) and drive_cfg.get("enabled", False))
    selected_noisy_methods = _normalize_display_string_list(
        settings.get("noisy_methods"),
        default=["cfqm4", "suzuki2"],
    )
    selected_noise_modes = _normalize_display_string_list(
        settings.get("noise_modes"),
        default=["ideal", "shots", "aer_noise"],
    )

    manifest_sections: list[tuple[str, list[tuple[str, Any]]]] = [
        (
            "Model and regime",
            [
                ("Model family", "Hubbard-Holstein"),
                ("L", settings.get("L")),
                ("Drive enabled", drive_enabled),
                ("Disable time dynamics", disable_time_dynamics),
                ("Noise modes", selected_noise_modes),
                ("Configured noisy method set", [report_method_label(m) for m in selected_noisy_methods]),
            ],
        ),
        (
            "Method chain",
            [
                (report_stage_label("warm_start"), "hh_hva_ptw"),
                (report_stage_label("adapt_pool_b"), "Pool B (UCCSD + HVA + PAOP_FULL)"),
                (report_stage_label("conventional_vqe"), "hh_hva_ptw"),
                ("Noiseless comparison methods", [report_method_label(name) for name in ("suzuki2", "magnus2", "cfqm4", "cfqm6")]),
            ],
        ),
        (
            "Core physical parameters",
            [
                ("t", settings.get("t")),
                ("U", settings.get("u")),
                ("dv", settings.get("dv")),
                ("omega0", settings.get("omega0")),
                ("g_ep", settings.get("g_ep")),
                ("n_ph_max", settings.get("n_ph_max")),
            ],
        ),
        (
            "Dynamics and audit settings",
            [
                ("trotter_steps", settings.get("trotter_steps")),
                ("num_times", settings.get("num_times")),
                ("shots", settings.get("shots")),
                ("Mitigation", settings.get("mitigation_config")),
                ("Symmetry mitigation", settings.get("symmetry_mitigation_config")),
                ("Hardcoded family", "deactivated (final-only dynamics)"),
            ],
        ),
    ]

    summary_sections: list[tuple[str, list[tuple[str, Any]]]] = [
        (
            "Headline robustness metrics",
            [
                (report_metric_label("final_delta_abs"), summary.get("final_delta_abs")),
                (report_metric_label("max_abs_delta"), summary.get("max_abs_delta")),
                (report_metric_label("max_abs_delta_over_stderr"), summary.get("max_abs_delta_over_stderr")),
                (report_metric_label("mean_abs_delta_over_stderr"), summary.get("mean_abs_delta_over_stderr")),
                (report_metric_label("noisy_audit_max_abs_delta"), summary.get("noisy_audit_max_abs_delta")),
            ],
        ),
        (
            "State-preparation checkpoints",
            [
                (f"{report_stage_label('warm_start')} |ΔE|", summary.get("warm_delta_abs")),
                (f"{report_stage_label('warm_start')} stop", summary.get("warm_stop_reason")),
                (f"{report_stage_label('adapt_pool_b')} |ΔE|", summary.get("adapt_delta_abs")),
                (f"{report_stage_label('adapt_pool_b')} stop", summary.get("adapt_stop_reason")),
                (f"{report_stage_label('conventional_vqe')} stop", summary.get("final_stop_reason")),
            ],
        ),
        (
            "Coverage and audit",
            [
                (report_metric_label("noisy_modes_completed"), f"{summary.get('noisy_modes_completed')} / {summary.get('noisy_modes_total')}"),
                (report_metric_label("noisy_method_modes_completed"), f"{summary.get('noisy_method_modes_completed')} / {summary.get('noisy_method_modes_total')}"),
                (report_metric_label("noisy_audit_modes_completed"), f"{summary.get('noisy_audit_modes_completed')} / {summary.get('noisy_audit_modes_total')}"),
                (report_metric_label("dynamics_benchmark_rows"), summary.get("dynamics_benchmark_rows")),
            ],
        ),
        (
            "Transition policy",
            [
                ("Warm → ADAPT", stage_pipeline.get("warm_start", {}).get("transition", {}).get("policy", {})),
                ("ADAPT → final replay", stage_pipeline.get("adapt_pool_b", {}).get("transition", {}).get("policy", {})),
            ],
        ),
    ]

    with PdfPages(str(pdf_path)) as pdf:
        render_manifest_overview_page(
            pdf,
            title=f"Hubbard-Holstein report — L={settings.get('L')}",
            experiment_statement=(
                "Driven HH robustness study with warm-start → ADAPT → final replay state preparation, "
                "followed by noiseless and noisy dynamics seeded from the final VQE state."
            ),
            sections=manifest_sections,
            notes=[
                "Warm / ADAPT / final are optimization checkpoints only.",
                "Dynamics and noise pages in this report propagate from the final VQE state only.",
                "The full command and formula atlas are appendix material.",
            ],
        )
        render_executive_summary_page(
            pdf,
            title="Executive summary",
            experiment_statement=(
                "Front-matter summary of prepared-state quality, noisy-minus-ideal effect size, and audit coverage."
            ),
            sections=summary_sections,
            notes=[
                f"Configured noisy method set: {', '.join(report_method_label(m) for m in selected_noisy_methods) or 'none'}.",
                "Magnus / CFQM stay in the noiseless comparison matrix; method-level coverage is recorded in the benchmark appendix.",
            ],
        )
        render_section_divider_page(
            pdf,
            title="State-preparation checkpoints",
            summary="Checkpoint pages report warm-start, ADAPT, and final replay quality before any propagated dynamics.",
            bullets=[
                "Transition traces and Pool-B composition.",
                "Warm / ADAPT / final are not separate propagated branches in this report path.",
            ],
        )

        # Stage transition slopes
        fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=False)
        fig.subplots_adjust(right=0.68)
        for ax, stage_key, title in [
            (axes[0], "warm_start", "Warm-start transition trace"),
            (axes[1], "adapt_pool_b", "ADAPT transition trace"),
        ]:
            trans = stage_pipeline.get(stage_key, {}).get("transition", {})
            d = np.asarray(trans.get("delta_abs_trace", []), dtype=float)
            s = np.asarray(trans.get("slope_trace", []), dtype=float)
            ax.set_title(title)
            if d.size > 0:
                ax.plot(np.arange(d.size), d, color="#1f77b4", linewidth=1.4, label="|DeltaE|")
                ax.set_ylabel("|DeltaE|")
            if s.size > 0:
                ax2 = ax.twinx()
                ax2.plot(np.arange(max(0, d.size - s.size), d.size), s, color="#d62728", linewidth=1.2, linestyle="--", label="slope")
                ax2.set_ylabel("slope")
            ax.grid(alpha=0.25)
        axes[1].set_xlabel("checkpoint index")
        _annotate_plot_with_equations(
            fig,
            eq_ids=["eq_delta_e", "eq_slope_switch"],
            equation_registry=equation_registry,
            plot_id="plot_stage_transition",
            plot_contracts=plot_contracts,
            style_legend_lines=branch_style_lines,
        )
        used_eq_ids.update(["eq_delta_e", "eq_slope_switch"])
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Pool B accounting table
        fig_tbl, ax_tbl = plt.subplots(figsize=(9.5, 5.0))
        fig_tbl.subplots_adjust(right=0.68)
        rows = [
            ["raw uccsd", str(pool_B.get("raw_sizes", {}).get("uccsd", ""))],
            ["raw hva", str(pool_B.get("raw_sizes", {}).get("hva", ""))],
            ["raw paop_full", str(pool_B.get("raw_sizes", {}).get("paop_full", ""))],
            ["dedup total", str(pool_B.get("dedup_total", ""))],
            ["presence uccsd", str(pool_B.get("dedup_source_presence_counts", {}).get("uccsd", ""))],
            ["presence hva", str(pool_B.get("dedup_source_presence_counts", {}).get("hva", ""))],
            ["presence paop_full", str(pool_B.get("dedup_source_presence_counts", {}).get("paop_full", ""))],
            ["overlap count", str(pool_B.get("overlap_count", ""))],
        ]
        render_compact_table(
            ax_tbl,
            title="Pool B Composition (strict union + dedup)",
            col_labels=["Metric", "Value"],
            rows=rows,
        )
        _annotate_plot_with_equations(
            fig_tbl,
            eq_ids=["eq_delta_e"],
            equation_registry=equation_registry,
            plot_id="plot_stage_transition",
            plot_contracts=plot_contracts,
            style_legend_lines=branch_style_lines,
        )
        used_eq_ids.update(["eq_delta_e"])
        fig_tbl.tight_layout()
        pdf.savefig(fig_tbl)
        plt.close(fig_tbl)

        warm_stage = stage_pipeline.get("warm_start", {}) if isinstance(stage_pipeline, dict) else {}
        adapt_stage = stage_pipeline.get("adapt_pool_b", {}) if isinstance(stage_pipeline, dict) else {}
        final_stage = stage_pipeline.get("conventional_vqe", {}) if isinstance(stage_pipeline, dict) else {}
        render_text_page(
            pdf,
            [
                "SECTION: RESULTS STAGE OPTIMIZATION SUMMARY",
                "",
                "Warm/ADAPT/final are optimization checkpoints only.",
                "They are not independently time-evolved in this report path.",
                "Dynamics/noise comparisons below are seeded from final VQE only.",
                "",
                (
                    f"{report_stage_label('warm_start')}: energy={warm_stage.get('energy')} "
                    f"delta_abs={warm_stage.get('delta_abs')} stop={warm_stage.get('stop_reason')}"
                ),
                (
                    f"{report_stage_label('adapt_pool_b')}: energy={adapt_stage.get('energy')} "
                    f"delta_abs={adapt_stage.get('delta_abs')} stop={adapt_stage.get('stop_reason')}"
                ),
                (
                    f"{report_stage_label('conventional_vqe')}: energy={final_stage.get('energy')} "
                    f"delta_abs={final_stage.get('delta_abs')} stop={final_stage.get('stop_reason')}"
                ),
                "",
                f"hardcoded_superset disabled: {bool(hardcoded_superset.get('disabled', False))}",
                f"reason: {hardcoded_superset.get('reason', '')}",
            ],
            fontsize=10,
        )

        render_section_divider_page(
            pdf,
            title="Dynamics from final replay state",
            summary="Noiseless comparison pages show exact reference versus final-state propagation with Suzuki, Magnus, and CFQM methods.",
            bullets=[
                "Warm / ADAPT / final checkpoints do not appear as propagated branches here.",
                "Method labels are presentation labels only; raw payload keys remain in sidecars.",
            ],
        )
        if disable_time_dynamics:
            render_text_page(
                pdf,
                [
                    "SECTION: RESULTS METHOD COMPARISON",
                    "",
                    "Time-dynamics propagation disabled by --disable-time-dynamics.",
                    "No Suzuki/Magnus/CFQM trajectory overlays are produced in this run.",
                    "Noisy outputs are reported via final-state t=0 audit pages.",
                ],
                fontsize=10,
            )

        # Noiseless overlays for static + drive (Magnus/CFQM extension)
        for profile_name in ("static", "drive"):
            pdata = noiseless.get(profile_name)
            if not isinstance(pdata, dict):
                continue
            times = np.asarray(pdata.get("times", []), dtype=float)
            if times.size == 0:
                continue
            ref = pdata.get("reference", {})
            e_ref = np.asarray(ref.get("energy_total_exact", []), dtype=float)

            fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)
            fig.subplots_adjust(right=0.68)
            axes[0].plot(times, e_ref, color="#111111", linewidth=2.0, label="exact reference")

            for name, color in [
                ("suzuki2", "#1f77b4"),
                ("magnus2", "#ff7f0e"),
                ("cfqm4", "#2ca02c"),
                ("cfqm6", "#d62728"),
            ]:
                m = pdata.get("methods", {}).get(name, {})
                traj = m.get("trajectory", [])
                if not traj:
                    continue
                e = np.asarray([float(r["energy_total_trotter"]) for r in traj], dtype=float)
                f = np.asarray([float(r["fidelity"]) for r in traj], dtype=float)
                axes[0].plot(times, e, linewidth=1.2, label=name, color=color)
                axes[1].plot(times, f, linewidth=1.2, label=name, color=color)

            axes[0].set_ylabel("Energy total")
            axes[0].set_title(f"Noiseless methods overlay ({profile_name})")
            axes[0].grid(alpha=0.25)
            axes[0].legend(fontsize=8, ncol=3)
            axes[1].set_ylabel("Subspace fidelity")
            axes[1].set_xlabel("Time")
            axes[1].grid(alpha=0.25)
            axes[1].legend(fontsize=8, ncol=3)
            _annotate_plot_with_equations(
                fig,
                eq_ids=["eq_energy_total", "eq_subspace_fidelity", "eq_suzuki2", "eq_magnus2", "eq_cfqm"],
                equation_registry=equation_registry,
                plot_id=f"plot_{profile_name}_magnus_cfqm_overlay",
                plot_contracts=plot_contracts,
                style_legend_lines=method_style_lines,
            )
            used_eq_ids.update(["eq_energy_total", "eq_subspace_fidelity", "eq_suzuki2", "eq_magnus2", "eq_cfqm"])
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # Method error page vs exact
            fig_err, axes_err = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)
            fig_err.subplots_adjust(right=0.68)
            for name, color in [
                ("suzuki2", "#1f77b4"),
                ("magnus2", "#ff7f0e"),
                ("cfqm4", "#2ca02c"),
                ("cfqm6", "#d62728"),
            ]:
                traj = pdata.get("methods", {}).get(name, {}).get("trajectory", [])
                if not traj:
                    continue
                e = np.asarray([float(r["energy_total_trotter"]) for r in traj], dtype=float)
                f = np.asarray([float(r["fidelity"]) for r in traj], dtype=float)
                axes_err[0].plot(times, np.abs(e - e_ref), color=color, linewidth=1.3, label=name)
                axes_err[1].plot(times, np.abs(1.0 - f), color=color, linewidth=1.3, label=name)
            axes_err[0].set_ylabel("|E_method-E_exact|")
            axes_err[0].grid(alpha=0.25)
            axes_err[0].legend(fontsize=8, ncol=3)
            axes_err[1].set_ylabel("|1-F_sub|")
            axes_err[1].set_xlabel("Time")
            axes_err[1].grid(alpha=0.25)
            axes_err[1].legend(fontsize=8, ncol=3)
            axes_err[0].set_title(f"{profile_name}: Magnus/CFQM error-to-exact")
            _annotate_plot_with_equations(
                fig_err,
                eq_ids=["eq_error_abs", "eq_subspace_fidelity", "eq_suzuki2", "eq_magnus2", "eq_cfqm"],
                equation_registry=equation_registry,
                plot_id=f"plot_{profile_name}_magnus_cfqm_error",
                plot_contracts=plot_contracts,
                style_legend_lines=method_style_lines,
            )
            used_eq_ids.update(["eq_error_abs", "eq_subspace_fidelity", "eq_suzuki2", "eq_magnus2", "eq_cfqm"])
            fig_err.tight_layout()
            pdf.savefig(fig_err)
            plt.close(fig_err)

        render_section_divider_page(
            pdf,
            title="Noisy-minus-ideal dynamics",
            summary="Noise pages compare the final-state noiseless baseline to selected noisy modes and show Δ(noisy-ideal) significance bands.",
            bullets=[
                f"Configured noisy method set: {', '.join(report_method_label(m) for m in selected_noisy_methods) or 'none'}.",
                f"Selected modes: {', '.join(selected_noise_modes) or 'none'}.",
                f"Shot budget: {int(settings.get('shots', 0))}.",
            ],
        )
        if disable_time_dynamics:
            render_text_page(
                pdf,
                [
                    "SECTION: RESULTS NOISE DETAILS",
                    "",
                    "Time-dynamics propagation disabled by --disable-time-dynamics.",
                    "Noise results below are final-state t=0 audits on the seeded conventional VQE state.",
                    "Each mode reports noisy/ideal means, propagated delta stderr, and significance metrics.",
                ],
                fontsize=10,
            )

        # Mandatory overlay pages: exact + noiseless + all available noisy modes.
        for profile_name in ("static", "drive"):
            nprof = noisy.get(profile_name)
            pprof = noiseless.get(profile_name)
            if not isinstance(nprof, dict) or not isinstance(pprof, dict):
                continue
            base_traj = pprof.get("methods", {}).get("suzuki2", {}).get("trajectory", [])
            if not base_traj:
                continue
            times = np.asarray([float(r["time"]) for r in base_traj], dtype=float)
            mode_order = list(selected_noise_modes)
            available_modes = [
                m for m in mode_order if bool(nprof.get("modes", {}).get(str(m), {}).get("success", False))
            ]
            unavailable_modes = [m for m in mode_order if m not in available_modes]

            # Energy overlay
            fig_eov, axes_eov = plt.subplots(2, 1, figsize=(8.5, 11.0), sharex=True)
            axes_eov[0].plot(
                times,
                np.asarray([float(r["energy_total_exact"]) for r in base_traj], dtype=float),
                color="#111111",
                linewidth=2.2,
                label="exact reference",
            )
            axes_eov[0].plot(
                times,
                np.asarray([float(r["energy_total_trotter"]) for r in base_traj], dtype=float),
                color="#1f77b4",
                linewidth=1.6,
                label="noiseless (final-seed Suzuki-2)",
            )
            axes_eov[1].plot(
                times,
                np.asarray([float(r["energy_static_trotter"]) for r in base_traj], dtype=float),
                color="#1f77b4",
                linewidth=1.6,
                label="noiseless (final-seed Suzuki-2)",
            )
            for mode in available_modes:
                color = {"ideal": "#ff7f0e", "shots": "#2ca02c", "aer_noise": "#d62728"}.get(str(mode), "#9467bd")
                traj = nprof.get("modes", {}).get(str(mode), {}).get("trajectory", [])
                if not traj:
                    continue
                axes_eov[0].plot(
                    times,
                    np.asarray([float(r["energy_total_noisy"]) for r in traj], dtype=float),
                    color=color,
                    linewidth=1.3,
                    label=f"noise mode={mode} measured",
                )
                axes_eov[1].plot(
                    times,
                    np.asarray([float(r["energy_static_noisy"]) for r in traj], dtype=float),
                    color=color,
                    linewidth=1.3,
                    label=f"noise mode={mode} measured",
                )
            axes_eov[0].set_title(f"{profile_name}: energy overlays")
            axes_eov[0].set_ylabel("E_total")
            axes_eov[1].set_ylabel("E_static")
            axes_eov[1].set_xlabel("time")
            axes_eov[0].grid(alpha=0.25)
            axes_eov[1].grid(alpha=0.25)
            axes_eov[0].legend(fontsize=8, ncol=2)
            axes_eov[1].legend(fontsize=8, ncol=2)
            _annotate_plot_with_equations(
                fig_eov,
                eq_ids=["eq_energy_total", "eq_energy_static", "eq_noisy_estimator"],
                equation_registry=equation_registry,
                plot_id=f"plot_{profile_name}_overlay_energy",
                plot_contracts=plot_contracts,
                style_legend_lines=noise_style_lines
                + [f"unavailable modes: {', '.join(unavailable_modes) if unavailable_modes else 'none'}"],
            )
            used_eq_ids.update(["eq_energy_total", "eq_energy_static", "eq_noisy_estimator"])
            fig_eov.tight_layout()
            pdf.savefig(fig_eov)
            plt.close(fig_eov)

            # Site-0 occupation overlay
            fig_oov, axes_oov = plt.subplots(2, 1, figsize=(8.5, 11.0), sharex=True)
            axes_oov[0].plot(
                times,
                np.asarray([float(r["n_up_site0_trotter_hva"]) for r in base_traj], dtype=float),
                color="#1f77b4",
                linewidth=1.5,
                label="noiseless (final-seed) up0",
            )
            axes_oov[1].plot(
                times,
                np.asarray([float(r["n_dn_site0_trotter_hva"]) for r in base_traj], dtype=float),
                color="#1f77b4",
                linewidth=1.5,
                label="noiseless (final-seed) dn0",
            )
            for mode in available_modes:
                color = {"ideal": "#ff7f0e", "shots": "#2ca02c", "aer_noise": "#d62728"}.get(str(mode), "#9467bd")
                traj = nprof.get("modes", {}).get(str(mode), {}).get("trajectory", [])
                if not traj:
                    continue
                axes_oov[0].plot(
                    times,
                    np.asarray([float(r["n_up_site0_noisy"]) for r in traj], dtype=float),
                    color=color,
                    linewidth=1.3,
                    label=f"{mode} measured up0",
                )
                axes_oov[1].plot(
                    times,
                    np.asarray([float(r["n_dn_site0_noisy"]) for r in traj], dtype=float),
                    color=color,
                    linewidth=1.3,
                    label=f"{mode} measured dn0",
                )
            axes_oov[0].set_title(f"{profile_name}: site-0 occupation overlays")
            axes_oov[0].set_ylabel("n_up(0)")
            axes_oov[1].set_ylabel("n_dn(0)")
            axes_oov[1].set_xlabel("time")
            axes_oov[0].grid(alpha=0.25)
            axes_oov[1].grid(alpha=0.25)
            axes_oov[0].legend(fontsize=8, ncol=2)
            axes_oov[1].legend(fontsize=8, ncol=2)
            _annotate_plot_with_equations(
                fig_oov,
                eq_ids=["eq_number_operator", "eq_total_density", "eq_noisy_estimator"],
                equation_registry=equation_registry,
                plot_id=f"plot_{profile_name}_overlay_occupancy",
                plot_contracts=plot_contracts,
                style_legend_lines=noise_style_lines
                + [f"unavailable modes: {', '.join(unavailable_modes) if unavailable_modes else 'none'}"],
            )
            used_eq_ids.update(["eq_number_operator", "eq_total_density", "eq_noisy_estimator"])
            fig_oov.tight_layout()
            pdf.savefig(fig_oov)
            plt.close(fig_oov)

            # Doublon / staggered overlay
            fig_dov, axes_dov = plt.subplots(2, 1, figsize=(8.5, 11.0), sharex=True)
            axes_dov[0].plot(
                times,
                np.asarray([float(r["doublon_trotter_hva"]) for r in base_traj], dtype=float),
                color="#1f77b4",
                linewidth=1.5,
                label="noiseless (final-seed) doublon",
            )
            axes_dov[1].plot(
                times,
                np.asarray([float(r["staggered_trotter_hva"]) for r in base_traj], dtype=float),
                color="#1f77b4",
                linewidth=1.5,
                label="noiseless (final-seed) staggered",
            )
            for mode in available_modes:
                color = {"ideal": "#ff7f0e", "shots": "#2ca02c", "aer_noise": "#d62728"}.get(str(mode), "#9467bd")
                traj = nprof.get("modes", {}).get(str(mode), {}).get("trajectory", [])
                if not traj:
                    continue
                axes_dov[0].plot(
                    times,
                    np.asarray([float(r["doublon_noisy"]) for r in traj], dtype=float),
                    color=color,
                    linewidth=1.3,
                    label=f"{mode} measured doublon",
                )
                axes_dov[1].plot(
                    times,
                    np.asarray([float(r["staggered_noisy"]) for r in traj], dtype=float),
                    color=color,
                    linewidth=1.3,
                    label=f"{mode} measured staggered",
                )
            axes_dov[0].set_title(f"{profile_name}: doublon/staggered overlays")
            axes_dov[0].set_ylabel("doublon")
            axes_dov[1].set_ylabel("staggered")
            axes_dov[1].set_xlabel("time")
            axes_dov[0].grid(alpha=0.25)
            axes_dov[1].grid(alpha=0.25)
            axes_dov[0].legend(fontsize=8, ncol=2)
            axes_dov[1].legend(fontsize=8, ncol=2)
            _annotate_plot_with_equations(
                fig_dov,
                eq_ids=["eq_doublon", "eq_staggered", "eq_noisy_estimator"],
                equation_registry=equation_registry,
                plot_id=f"plot_{profile_name}_overlay_doublon_staggered",
                plot_contracts=plot_contracts,
                style_legend_lines=noise_style_lines
                + [f"unavailable modes: {', '.join(unavailable_modes) if unavailable_modes else 'none'}"],
            )
            used_eq_ids.update(["eq_doublon", "eq_staggered", "eq_noisy_estimator"])
            fig_dov.tight_layout()
            pdf.savefig(fig_dov)
            plt.close(fig_dov)

        # Noisy vs noiseless Suzuki full per-mode pages
        for profile_name in ("static", "drive"):
            nprof = noisy.get(profile_name)
            pprof = noiseless.get(profile_name)
            if not isinstance(nprof, dict) or not isinstance(pprof, dict):
                continue
            base_traj = pprof.get("methods", {}).get("suzuki2", {}).get("trajectory", [])
            if not base_traj:
                continue

            times = np.asarray([float(r["time"]) for r in base_traj], dtype=float)
            e_base = np.asarray([float(r["energy_total_trotter"]) for r in base_traj], dtype=float)
            s_base = np.asarray([float(r["energy_static_trotter"]) for r in base_traj], dtype=float)
            d_base = np.asarray([float(r["doublon_trotter_hva"]) for r in base_traj], dtype=float)
            stg_base = np.asarray([float(r["staggered_trotter_hva"]) for r in base_traj], dtype=float)
            e_ref = np.asarray([float(r["energy_total_exact"]) for r in base_traj], dtype=float)
            up_base = np.asarray([float(r["n_up_site0_trotter_hva"]) for r in base_traj], dtype=float)
            dn_base = np.asarray([float(r["n_dn_site0_trotter_hva"]) for r in base_traj], dtype=float)

            mode_order = list(selected_noise_modes)
            for mode in mode_order:
                color = {"ideal": "#ff7f0e", "shots": "#2ca02c", "aer_noise": "#d62728"}.get(str(mode), "#9467bd")
                mdata = nprof.get("modes", {}).get(str(mode), {})
                if not bool(mdata.get("success", False)):
                    render_text_page(
                        pdf,
                        [
                            f"SECTION: NOISE MODE UNAVAILABLE ({profile_name}/{mode})",
                            "",
                            f"Reason: {mdata.get('reason', mdata.get('error', 'unknown'))}",
                            f"Diagnostics: {json.dumps(mdata, indent=2)}",
                        ],
                        fontsize=9,
                    )
                    continue
                traj = mdata.get("trajectory", [])
                if not traj:
                    continue
                e_tot_n = np.asarray([float(r["energy_total_noisy"]) for r in traj], dtype=float)
                e_tot_i = np.asarray([float(r["energy_total_ideal"]) for r in traj], dtype=float)
                e_sta_n = np.asarray([float(r["energy_static_noisy"]) for r in traj], dtype=float)
                e_sta_i = np.asarray([float(r["energy_static_ideal"]) for r in traj], dtype=float)
                e_tot_d = np.asarray([float(r["energy_total_delta_noisy_minus_ideal"]) for r in traj], dtype=float)
                e_tot_d_se = np.asarray(
                    [float(r.get("energy_total_delta_noisy_minus_ideal_stderr", 0.0)) for r in traj],
                    dtype=float,
                )
                e_sta_d = np.asarray([float(r["energy_static_delta_noisy_minus_ideal"]) for r in traj], dtype=float)
                e_sta_d_se = np.asarray(
                    [float(r.get("energy_static_delta_noisy_minus_ideal_stderr", 0.0)) for r in traj],
                    dtype=float,
                )
                up_n = np.asarray([float(r["n_up_site0_noisy"]) for r in traj], dtype=float)
                up_i = np.asarray([float(r["n_up_site0_ideal"]) for r in traj], dtype=float)
                up_d = np.asarray([float(r["n_up_site0_delta_noisy_minus_ideal"]) for r in traj], dtype=float)
                up_d_se = np.asarray(
                    [float(r.get("n_up_site0_delta_noisy_minus_ideal_stderr", 0.0)) for r in traj], dtype=float
                )
                dn_n = np.asarray([float(r["n_dn_site0_noisy"]) for r in traj], dtype=float)
                dn_i = np.asarray([float(r["n_dn_site0_ideal"]) for r in traj], dtype=float)
                dn_d = np.asarray([float(r["n_dn_site0_delta_noisy_minus_ideal"]) for r in traj], dtype=float)
                dn_d_se = np.asarray(
                    [float(r.get("n_dn_site0_delta_noisy_minus_ideal_stderr", 0.0)) for r in traj], dtype=float
                )
                dbl_n = np.asarray([float(r["doublon_noisy"]) for r in traj], dtype=float)
                dbl_i = np.asarray([float(r["doublon_ideal"]) for r in traj], dtype=float)
                dbl_d = np.asarray([float(r["doublon_delta_noisy_minus_ideal"]) for r in traj], dtype=float)
                dbl_d_se = np.asarray(
                    [float(r.get("doublon_delta_noisy_minus_ideal_stderr", 0.0)) for r in traj], dtype=float
                )
                stg_n = np.asarray([float(r["staggered_noisy"]) for r in traj], dtype=float)
                stg_i = np.asarray([float(r["staggered_ideal"]) for r in traj], dtype=float)
                stg_d = np.asarray([float(r["staggered_delta_noisy_minus_ideal"]) for r in traj], dtype=float)
                stg_d_se = np.asarray(
                    [float(r.get("staggered_delta_noisy_minus_ideal_stderr", 0.0)) for r in traj], dtype=float
                )
                mode_unc = mdata.get("delta_uncertainty", {})
                energy_unc = (
                    mode_unc.get("energy_total", {})
                    if isinstance(mode_unc, dict) and isinstance(mode_unc.get("energy_total", {}), dict)
                    else {}
                )
                delta_stats = (
                    f"max|ΔE_total|={float(np.max(np.abs(e_tot_d))):.3e}; "
                    f"RMS|ΔE_total|={float(np.sqrt(np.mean(e_tot_d ** 2))):.3e}; "
                    f"max|ΔD|={float(np.max(np.abs(dbl_d))):.3e}; "
                    f"max|Δ|/stderr={float(energy_unc.get('max_abs_delta_over_stderr', float('nan'))):.3e}"
                )
                noise_cfg_line = _noise_config_caption(settings, str(mode))

                # Energy page
                fig_e, axes_e = plt.subplots(3, 1, figsize=(10.8, 9.0), sharex=True)
                fig_e.subplots_adjust(hspace=0.28)
                axes_e[0].plot(times, e_ref, color="#111111", linewidth=1.8, label="exact total")
                axes_e[0].plot(times, e_base, color="#1f77b4", linewidth=1.4, label="noiseless (final-seed Suzuki-2) total")
                axes_e[0].plot(times, e_tot_n, color=color, linewidth=1.2, label=f"noisy measured (mode={mode}) total")
                axes_e[0].plot(
                    times,
                    e_tot_i,
                    color=color,
                    linewidth=1.0,
                    linestyle="--",
                    label=f"ideal reference for mode={mode} total",
                )
                axes_e[0].set_ylabel("E_total")
                axes_e[0].grid(alpha=0.25)
                axes_e[0].legend(fontsize=7, ncol=2)
                axes_e[1].plot(times, s_base, color="#1f77b4", linewidth=1.4, label="noiseless (final-seed Suzuki-2) static")
                axes_e[1].plot(times, e_sta_n, color=color, linewidth=1.2, label=f"noisy measured (mode={mode}) static")
                axes_e[1].plot(
                    times,
                    e_sta_i,
                    color=color,
                    linewidth=1.0,
                    linestyle="--",
                    label=f"ideal reference for mode={mode} static",
                )
                axes_e[1].set_ylabel("E_static")
                axes_e[1].grid(alpha=0.25)
                axes_e[1].legend(fontsize=7, ncol=2)
                axes_e[2].plot(times, e_tot_d, color=color, linewidth=1.2, label="Δ(noisy-ideal) total")
                axes_e[2].plot(times, e_sta_d, color=color, linewidth=1.2, linestyle="--", label="Δ(noisy-ideal) static")
                axes_e[2].fill_between(
                    times,
                    e_tot_d - (2.0 * e_tot_d_se),
                    e_tot_d + (2.0 * e_tot_d_se),
                    color=color,
                    alpha=0.16,
                    linewidth=0.0,
                    label="±2 stderr (total)",
                )
                axes_e[2].fill_between(
                    times,
                    e_sta_d - (2.0 * e_sta_d_se),
                    e_sta_d + (2.0 * e_sta_d_se),
                    color=color,
                    alpha=0.10,
                    linewidth=0.0,
                    label="±2 stderr (static)",
                )
                axes_e[2].axhline(0.0, color="#111111", linewidth=0.8, linestyle=":", label="zero baseline")
                axes_e[2].set_ylabel("Δ(noisy-ideal) E")
                axes_e[2].set_xlabel("time")
                axes_e[2].grid(alpha=0.25)
                axes_e[2].legend(fontsize=7, ncol=2)
                fig_e.suptitle(f"{profile_name}/{mode}: noisy energy audit")
                _annotate_plot_with_equations(
                    fig_e,
                    eq_ids=["eq_energy_static", "eq_energy_total", "eq_noisy_estimator", "eq_noisy_delta"],
                    equation_registry=equation_registry,
                    plot_id=f"plot_{profile_name}_{mode}_energy",
                    plot_contracts=plot_contracts,
                    style_legend_lines=noise_style_lines + [noise_cfg_line, delta_stats],
                )
                used_eq_ids.update(["eq_energy_static", "eq_energy_total", "eq_noisy_estimator", "eq_noisy_delta"])
                fig_e.tight_layout()
                pdf.savefig(fig_e)
                plt.close(fig_e)

                # Occupation page
                fig_o, axes_o = plt.subplots(3, 1, figsize=(10.8, 9.0), sharex=True)
                fig_o.subplots_adjust(hspace=0.28)
                axes_o[0].plot(times, up_base, color="#1f77b4", linewidth=1.3, label="noiseless (final-seed) up0")
                axes_o[0].plot(times, up_n, color=color, linewidth=1.2, label=f"noisy measured (mode={mode}) up0")
                axes_o[0].plot(
                    times,
                    up_i,
                    color=color,
                    linewidth=1.0,
                    linestyle="--",
                    label=f"ideal reference for mode={mode} up0",
                )
                axes_o[0].set_ylabel("n_up(0)")
                axes_o[0].grid(alpha=0.25)
                axes_o[0].legend(fontsize=7, ncol=2)
                axes_o[1].plot(times, dn_base, color="#1f77b4", linewidth=1.3, label="noiseless (final-seed) dn0")
                axes_o[1].plot(times, dn_n, color=color, linewidth=1.2, label=f"noisy measured (mode={mode}) dn0")
                axes_o[1].plot(
                    times,
                    dn_i,
                    color=color,
                    linewidth=1.0,
                    linestyle="--",
                    label=f"ideal reference for mode={mode} dn0",
                )
                axes_o[1].set_ylabel("n_dn(0)")
                axes_o[1].grid(alpha=0.25)
                axes_o[1].legend(fontsize=7, ncol=2)
                axes_o[2].plot(times, up_d, color=color, linewidth=1.2, label="Δ(noisy-ideal) up0")
                axes_o[2].plot(times, dn_d, color=color, linewidth=1.2, linestyle="--", label="Δ(noisy-ideal) dn0")
                axes_o[2].fill_between(
                    times,
                    up_d - (2.0 * up_d_se),
                    up_d + (2.0 * up_d_se),
                    color=color,
                    alpha=0.16,
                    linewidth=0.0,
                    label="±2 stderr (up0)",
                )
                axes_o[2].fill_between(
                    times,
                    dn_d - (2.0 * dn_d_se),
                    dn_d + (2.0 * dn_d_se),
                    color=color,
                    alpha=0.10,
                    linewidth=0.0,
                    label="±2 stderr (dn0)",
                )
                axes_o[2].axhline(0.0, color="#111111", linewidth=0.8, linestyle=":", label="zero baseline")
                axes_o[2].set_ylabel("Δ(noisy-ideal) n(0)")
                axes_o[2].set_xlabel("time")
                axes_o[2].grid(alpha=0.25)
                axes_o[2].legend(fontsize=7, ncol=2)
                fig_o.suptitle(f"{profile_name}/{mode}: site-0 occupation audit")
                _annotate_plot_with_equations(
                    fig_o,
                    eq_ids=["eq_number_operator", "eq_total_density", "eq_noisy_estimator", "eq_noisy_delta"],
                    equation_registry=equation_registry,
                    plot_id=f"plot_{profile_name}_{mode}_occupancy",
                    plot_contracts=plot_contracts,
                    style_legend_lines=noise_style_lines + [noise_cfg_line, delta_stats],
                )
                used_eq_ids.update(["eq_number_operator", "eq_total_density", "eq_noisy_estimator", "eq_noisy_delta"])
                fig_o.tight_layout()
                pdf.savefig(fig_o)
                plt.close(fig_o)

                # Doublon/staggered page
                fig_d, axes_d = plt.subplots(3, 1, figsize=(10.8, 9.0), sharex=True)
                fig_d.subplots_adjust(hspace=0.28)
                axes_d[0].plot(times, d_base, color="#1f77b4", linewidth=1.3, label="noiseless (final-seed) doublon")
                axes_d[0].plot(times, dbl_n, color=color, linewidth=1.2, label=f"noisy measured (mode={mode}) doublon")
                axes_d[0].plot(
                    times,
                    dbl_i,
                    color=color,
                    linewidth=1.0,
                    linestyle="--",
                    label=f"ideal reference for mode={mode} doublon",
                )
                axes_d[0].set_ylabel("doublon")
                axes_d[0].grid(alpha=0.25)
                axes_d[0].legend(fontsize=7, ncol=2)
                axes_d[1].plot(times, stg_base, color="#1f77b4", linewidth=1.3, label="noiseless (final-seed) staggered")
                axes_d[1].plot(times, stg_n, color=color, linewidth=1.2, label=f"noisy measured (mode={mode}) staggered")
                axes_d[1].plot(
                    times,
                    stg_i,
                    color=color,
                    linewidth=1.0,
                    linestyle="--",
                    label=f"ideal reference for mode={mode} staggered",
                )
                axes_d[1].set_ylabel("staggered")
                axes_d[1].grid(alpha=0.25)
                axes_d[1].legend(fontsize=7, ncol=2)
                axes_d[2].plot(times, dbl_d, color=color, linewidth=1.2, label="Δ(noisy-ideal) doublon")
                axes_d[2].plot(times, stg_d, color=color, linewidth=1.2, linestyle="--", label="Δ(noisy-ideal) staggered")
                axes_d[2].fill_between(
                    times,
                    dbl_d - (2.0 * dbl_d_se),
                    dbl_d + (2.0 * dbl_d_se),
                    color=color,
                    alpha=0.16,
                    linewidth=0.0,
                    label="±2 stderr (doublon)",
                )
                axes_d[2].fill_between(
                    times,
                    stg_d - (2.0 * stg_d_se),
                    stg_d + (2.0 * stg_d_se),
                    color=color,
                    alpha=0.10,
                    linewidth=0.0,
                    label="±2 stderr (staggered)",
                )
                axes_d[2].axhline(0.0, color="#111111", linewidth=0.8, linestyle=":", label="zero baseline")
                axes_d[2].set_ylabel("Δ(noisy-ideal)")
                axes_d[2].set_xlabel("time")
                axes_d[2].grid(alpha=0.25)
                axes_d[2].legend(fontsize=7, ncol=2)
                fig_d.suptitle(f"{profile_name}/{mode}: doublon & staggered audit")
                _annotate_plot_with_equations(
                    fig_d,
                    eq_ids=["eq_doublon", "eq_staggered", "eq_noisy_estimator", "eq_noisy_delta"],
                    equation_registry=equation_registry,
                    plot_id=f"plot_{profile_name}_{mode}_doublon_staggered",
                    plot_contracts=plot_contracts,
                    style_legend_lines=noise_style_lines + [noise_cfg_line, delta_stats],
                )
                used_eq_ids.update(["eq_doublon", "eq_staggered", "eq_noisy_estimator", "eq_noisy_delta"])
                fig_d.tight_layout()
                pdf.savefig(fig_d)
                plt.close(fig_d)

        noisy_audit = payload.get("noisy_final_audit", {}).get("profiles", {})
        if bool(settings.get("disable_time_dynamics", False)) and isinstance(noisy_audit, dict):
            for profile_name in ("static", "drive"):
                prof = noisy_audit.get(profile_name, {})
                if not isinstance(prof, dict):
                    continue
                modes = prof.get("modes", {}) if isinstance(prof.get("modes", {}), dict) else {}
                for mode_name, mode_data in modes.items():
                    if not bool(isinstance(mode_data, dict) and mode_data.get("success", False)):
                        render_text_page(
                            pdf,
                            [
                                f"SECTION: NOISE MODE UNAVAILABLE ({profile_name}/{mode_name})",
                                "",
                                f"Reason: {mode_data.get('reason', mode_data.get('error', 'unknown'))}",
                                f"Diagnostics: {json.dumps(mode_data, indent=2)}",
                            ],
                            fontsize=9,
                        )
                        continue
                    obs = mode_data.get("final_observables", {}) if isinstance(mode_data, dict) else {}
                    noise_cfg = mode_data.get("noise_config", {}) if isinstance(mode_data, dict) else {}
                    cfg_line = (
                        "noise_config: "
                        f"mode={noise_cfg.get('noise_mode')}, "
                        f"shots={noise_cfg.get('shots')}, "
                        f"oracle_repeats={noise_cfg.get('oracle_repeats')}, "
                        f"oracle_aggregate={noise_cfg.get('oracle_aggregate')}, "
                        f"mitigation={noise_cfg.get('mitigation', {}).get('mode', 'none')}"
                    )
                    rows = []
                    for obs_key in [
                        "energy_static",
                        "energy_total",
                        "n_up_site0",
                        "n_dn_site0",
                        "doublon",
                        "staggered",
                    ]:
                        rec = obs.get(obs_key, {}) if isinstance(obs.get(obs_key, {}), dict) else {}
                        rows.append(
                            [
                                str(obs_key),
                                f"{float(rec.get('noisy_mean', float('nan'))):.8f}",
                                f"{float(rec.get('noisy_stderr', float('nan'))):.3e}",
                                f"{float(rec.get('ideal_mean', float('nan'))):.8f}",
                                f"{float(rec.get('ideal_stderr', float('nan'))):.3e}",
                                f"{float(rec.get('delta_mean', float('nan'))):.3e}",
                                f"{float(rec.get('delta_stderr', float('nan'))):.3e}",
                            ]
                        )
                    fig_tbl, ax_tbl = plt.subplots(figsize=(12.0, 6.0))
                    render_compact_table(
                        ax_tbl,
                        title=f"Final-state noisy audit (t=0) profile={profile_name}, mode={mode_name}",
                        col_labels=[
                            "Observable",
                            "Noisy mean",
                            "Noisy stderr",
                            "Ideal mean",
                            "Ideal stderr",
                            "Delta",
                            "Delta stderr",
                        ],
                        rows=rows,
                        fontsize=8,
                    )
                    fig_tbl.text(0.01, 0.01, cfg_line, fontsize=8, ha="left", va="bottom")
                    fig_tbl.tight_layout()
                    pdf.savefig(fig_tbl)
                    plt.close(fig_tbl)

        # Drive waveform diagnostics
        drive_cfg = settings.get("drive_profile", {})
        if isinstance(drive_cfg, dict) and bool(drive_cfg.get("enabled", False)):
            times = np.linspace(0.0, float(settings.get("t_final", 0.0)), int(settings.get("num_times", 1)))
            waveform = evaluate_drive_waveform(times, drive_cfg, float(drive_cfg.get("A", 0.0)))
            fig, ax = plt.subplots(figsize=(10.5, 3.8))
            fig.subplots_adjust(right=0.68)
            ax.plot(times, waveform, color="#111111", linewidth=1.6)
            ax.set_title("Drive waveform A*sin(omega t + phi)*exp(-(t+t0)^2/(2 tbar^2))")
            ax.set_xlabel("Time")
            ax.set_ylabel("drive scalar")
            ax.grid(alpha=0.25)
            _annotate_plot_with_equations(
                fig,
                eq_ids=["eq_drive_waveform", "eq_h_drive_density"],
                equation_registry=equation_registry,
                plot_id="plot_drive_waveform",
                plot_contracts={"plot_drive_waveform": {"x": "time", "y": ["drive scalar"], "source": ["settings.drive_profile"], "notes": "Drive scalar evaluation grid."}},
                style_legend_lines=method_style_lines,
            )
            used_eq_ids.update(["eq_drive_waveform", "eq_h_drive_density"])
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        render_section_divider_page(
            pdf,
            title="Benchmark and audit appendix",
            summary="Benchmark tables and diagnostics are separated from the main science pages so cost/error audit does not crowd the headline results.",
            bullets=[
                "Proxy costs and runtime totals.",
                "Diagnostics, equation definitions, plot contracts, and full command.",
            ],
        )
        benchmark_rows = payload.get("dynamics_benchmarks", {}).get("rows", [])
        if isinstance(benchmark_rows, list) and benchmark_rows:
            fig = plt.figure(figsize=(12.0, 7.0))
            ax = fig.add_subplot(111)
            headers = [
                "profile",
                "method",
                "mode",
                "cx",
                "term_exp",
                "sq",
                "depth",
                "wall_s",
                "oracle_s",
                "max|Δ|",
                "max|Δ|/stderr",
                "mean|Δ|/stderr",
            ]
            rows: list[list[str]] = []
            for rec in benchmark_rows:
                if not isinstance(rec, dict):
                    continue
                rows.append(
                    [
                        str(rec.get("profile", "")),
                        str(rec.get("method", "")),
                        str(rec.get("mode", "")),
                        f"{int(rec.get('cx_proxy_total', 0))}",
                        f"{int(rec.get('term_exp_count_total', 0))}",
                        f"{int(rec.get('sq_proxy_total', 0))}",
                        f"{int(rec.get('depth_proxy_total', 0))}",
                        f"{float(rec.get('wall_total_s', float('nan'))):.3f}",
                        f"{float(rec.get('oracle_eval_s_total', float('nan'))):.3f}",
                        f"{float(rec.get('max_abs_delta', float('nan'))):.3e}",
                        f"{float(rec.get('max_abs_delta_over_stderr', float('nan'))):.3e}",
                        f"{float(rec.get('mean_abs_delta_over_stderr', float('nan'))):.3e}",
                    ]
                )
            if rows:
                render_compact_table(
                    ax,
                    title="Noisy dynamics benchmark (Trotter vs CFQM under noise)",
                    col_labels=headers,
                    rows=rows,
                    fontsize=7,
                )
                _annotate_plot_with_equations(
                    fig,
                    eq_ids=["eq_proxy_cost", "eq_runtime_bench"],
                    equation_registry=equation_registry,
                    plot_id="plot_noisy_benchmark_table",
                    plot_contracts=plot_contracts,
                    style_legend_lines=method_style_lines,
                )
                used_eq_ids.update(["eq_proxy_cost", "eq_runtime_bench"])
                fig.tight_layout()
                pdf.savefig(fig)
            plt.close(fig)

        diag_lines = [
            "Diagnostics and availability",
            "",
            f"Comparisons keys: {list(comparisons.keys())}",
            f"Noisy mode diagnostics: {json.dumps(diagnostics.get('noisy_mode_diagnostics', {}), indent=2)}",
            f"Noisy dynamics benchmark rows: {len(payload.get('dynamics_benchmarks', {}).get('rows', []))}",
            "",
            "Metric formulas:",
            "  energy_static(t) = <psi(t)|H_static|psi(t)>",
            "  energy_total(t)  = <psi(t)|H_static + H_drive(t)|psi(t)>",
            "  n_up_site0 = <n_{0,up}> , n_dn_site0 = <n_{0,dn}>",
            "  doublon(site0) = <n_{0,up} n_{0,dn}>",
            "  staggered = (1/L) * Sum_j (-1)^j <n_j>",
            "",
            f"equation_registry_count: {len(equation_registry)}",
            f"plot_contract_count: {len(plot_contracts)}",
        ]
        render_text_page(pdf, diag_lines, fontsize=9)

        render_text_page(
            pdf,
            [
                "SECTION: APPENDIX DEFINITIONS USED",
                "",
                "Short equation definitions used in result pages.",
            ],
            fontsize=10,
        )
        used_keys = sorted(used_eq_ids)
        if len(used_keys) < 8:
            used_keys = sorted(list(equation_registry.keys()))[:24]
        _render_formula_atlas(
            pdf,
            equation_registry=equation_registry,
            title="SECTION: APPENDIX DEFINITIONS USED",
            keys=used_keys,
            include_sources=False,
        )

        _render_formula_atlas(
            pdf,
            equation_registry=equation_registry,
            title="SECTION: APPENDIX FORMULA ATLAS",
            keys=None,
            include_sources=True,
        )

        render_text_page(
            pdf,
            [
                "SECTION: APPENDIX PLOT CONTRACTS",
                "",
                "Full machine-readable plot contract map.",
            ],
            fontsize=10,
        )
        contract_lines: list[str] = ["SECTION: APPENDIX PLOT CONTRACTS", ""]
        for idx, key in enumerate(sorted(plot_contracts.keys())):
            rec = plot_contracts.get(key, {})
            contract_lines.append(f"[{key}]")
            contract_lines.append(f"  x: {rec.get('x', '')}")
            contract_lines.append(f"  y: {rec.get('y', '')}")
            contract_lines.append(f"  source: {rec.get('source', '')}")
            contract_lines.append(f"  notes: {rec.get('notes', '')}")
            contract_lines.append("")
            if (idx + 1) % 8 == 0:
                render_text_page(pdf, contract_lines, fontsize=8, line_spacing=0.024, max_line_width=140)
                contract_lines = [f"SECTION: APPENDIX PLOT CONTRACTS (cont. {idx + 1})", ""]
        if contract_lines:
            render_text_page(pdf, contract_lines, fontsize=8, line_spacing=0.024, max_line_width=140)

        render_command_page(
            pdf,
            str(payload.get("run_command", "")),
            script_name="pipelines/exact_bench/hh_noise_robustness_seq_report.py",
        )


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def _enforce_defaults_and_minimums(args: argparse.Namespace) -> argparse.Namespace:
    key = (int(args.L), int(args.n_ph_max))
    minimum = dict(_HH_MINIMUMS.get(key, _HH_MINIMUMS[(2, 1)]))

    if args.warm_reps is None:
        args.warm_reps = int(minimum["reps"])
    if args.warm_restarts is None:
        args.warm_restarts = int(minimum["restarts"])
    if args.warm_maxiter is None:
        args.warm_maxiter = int(minimum["maxiter"])

    if args.final_reps is None:
        args.final_reps = int(minimum["reps"])
    if args.final_restarts is None:
        args.final_restarts = int(minimum["restarts"])
    if args.final_maxiter is None:
        args.final_maxiter = int(minimum["maxiter"])

    if args.trotter_steps is None:
        args.trotter_steps = int(minimum["trotter_steps"])

    if args.warm_method is None:
        args.warm_method = str(minimum["method"])
    if args.final_method is None:
        args.final_method = str(minimum["method"])

    if args.num_times is None:
        args.num_times = 101
    if args.t_final is None:
        args.t_final = 10.0

    if not bool(args.smoke_test_intentionally_weak):
        checks = {
            "trotter_steps": int(args.trotter_steps) >= int(minimum["trotter_steps"]),
            "warm_reps": int(args.warm_reps) >= int(minimum["reps"]),
            "warm_restarts": int(args.warm_restarts) >= int(minimum["restarts"]),
            "warm_maxiter": int(args.warm_maxiter) >= int(minimum["maxiter"]),
            "final_reps": int(args.final_reps) >= int(minimum["reps"]),
            "final_restarts": int(args.final_restarts) >= int(minimum["restarts"]),
            "final_maxiter": int(args.final_maxiter) >= int(minimum["maxiter"]),
        }
        failed = [k for k, ok in checks.items() if not bool(ok)]
        if failed:
            raise ValueError(
                "Under-parameterized report run rejected by AGENTS minimum table. "
                f"Failed fields: {failed}. Minimums: {minimum}. "
                "Pass --smoke-test-intentionally-weak only for explicit smoke tests."
            )

    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "HH noise robustness comprehensive report: warm-start -> ADAPT PoolB -> conventional VQE, "
            "with noiseless Magnus/Trotter matrix and noisy method matrix."
        )
    )

    p.add_argument("--L", type=int, default=2)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--u", type=float, default=2.0)
    p.add_argument("--dv", type=float, default=0.0)
    p.add_argument("--omega0", type=float, default=1.0)
    p.add_argument("--g-ep", type=float, default=1.0)
    p.add_argument("--n-ph-max", type=int, default=1)
    p.add_argument("--boson-encoding", choices=["binary"], default="binary")
    p.add_argument("--ordering", choices=["blocked", "interleaved"], default="blocked")
    p.add_argument("--boundary", choices=["periodic", "open"], default="periodic")

    p.add_argument("--warm-reps", type=int, default=None)
    p.add_argument("--warm-restarts", type=int, default=None)
    p.add_argument("--warm-maxiter", type=int, default=None)
    p.add_argument("--warm-method", choices=["COBYLA", "SLSQP", "L-BFGS-B", "Powell", "Nelder-Mead"], default=None)
    p.add_argument("--warm-seed", type=int, default=7)

    p.add_argument("--window-k", type=int, default=5)
    p.add_argument("--slope-epsilon", type=float, default=5e-5)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min-points-before-switch", type=int, default=8)

    p.set_defaults(adapt_allow_repeats=False)
    p.add_argument("--adapt-allow-repeats", dest="adapt_allow_repeats", action="store_true")
    p.add_argument("--adapt-no-repeats", dest="adapt_allow_repeats", action="store_false")
    p.add_argument("--adapt-max-depth", type=int, default=40)
    p.add_argument("--adapt-maxiter", type=int, default=600)
    p.add_argument("--adapt-eps-grad", type=float, default=1e-6)
    p.add_argument("--adapt-eps-energy", type=float, default=1e-8)
    p.add_argument("--adapt-seed", type=int, default=11)

    p.add_argument("--paop-r", type=int, default=1)
    p.add_argument("--paop-split-paulis", action="store_true")
    p.add_argument("--paop-prune-eps", type=float, default=0.0)
    p.add_argument("--paop-normalization", choices=["none", "fro", "maxcoeff"], default="none")

    p.add_argument("--final-reps", type=int, default=None)
    p.add_argument("--final-restarts", type=int, default=None)
    p.add_argument("--final-maxiter", type=int, default=None)
    p.add_argument("--final-method", choices=["COBYLA", "SLSQP", "L-BFGS-B", "Powell", "Nelder-Mead"], default=None)
    p.add_argument("--final-seed", type=int, default=19)

    p.add_argument("--t-final", type=float, default=None)
    p.add_argument("--num-times", type=int, default=None)
    p.add_argument("--trotter-steps", type=int, default=None)
    p.add_argument("--exact-steps-multiplier", type=int, default=2)
    p.add_argument("--fidelity-subspace-energy-tol", type=float, default=1e-9)
    p.add_argument("--cfqm-stage-exp", choices=["expm_multiply_sparse", "dense_expm", "pauli_suzuki2"], default="expm_multiply_sparse")
    p.add_argument("--cfqm-coeff-drop-abs-tol", type=float, default=0.0)
    p.add_argument("--cfqm-normalize", action="store_true")
    p.add_argument("--disable-time-dynamics", action="store_true")

    p.add_argument("--include-drive-profile", action="store_true", default=True)
    p.add_argument("--drive-A", type=float, default=0.6)
    p.add_argument("--drive-omega", type=float, default=2.0)
    p.add_argument("--drive-tbar", type=float, default=2.5)
    p.add_argument("--drive-phi", type=float, default=0.0)
    p.add_argument("--drive-pattern", choices=["staggered", "dimer_bias", "custom"], default="staggered")
    p.add_argument("--drive-custom-s", type=str, default=None)
    p.add_argument("--drive-include-identity", action="store_true")
    p.add_argument("--drive-time-sampling", choices=["midpoint", "left", "right"], default="midpoint")
    p.add_argument("--drive-t0", type=float, default=0.0)

    p.add_argument("--noise-modes", type=str, default="ideal,shots,aer_noise")
    p.add_argument("--noisy-methods", type=str, default="cfqm4,suzuki2")
    p.add_argument("--benchmark-active-coeff-tol", type=float, default=1e-12)
    p.add_argument("--shots", type=int, default=2048)
    p.add_argument("--oracle-repeats", type=int, default=4)
    p.add_argument("--oracle-aggregate", choices=["mean", "median"], default="mean")
    p.add_argument("--mitigation", choices=["none", "readout", "zne", "dd"], default="none")
    p.add_argument(
        "--symmetry-mitigation-mode",
        choices=["off", "verify_only", "postselect_diag_v1", "projector_renorm_v1"],
        default="off",
    )
    p.add_argument("--zne-scales", type=str, default=None)
    p.add_argument("--dd-sequence", type=str, default=None)
    p.add_argument("--noise-seed", type=int, default=7)
    p.add_argument("--backend-name", type=str, default="FakeManilaV2")
    p.add_argument("--use-fake-backend", action="store_true")
    p.set_defaults(allow_aer_fallback=True)
    p.add_argument("--allow-aer-fallback", dest="allow_aer_fallback", action="store_true")
    p.add_argument("--no-allow-aer-fallback", dest="allow_aer_fallback", action="store_false")
    p.set_defaults(omp_shm_workaround=True)
    p.add_argument("--omp-shm-workaround", dest="omp_shm_workaround", action="store_true")
    p.add_argument("--no-omp-shm-workaround", dest="omp_shm_workaround", action="store_false")
    p.add_argument("--noisy-mode-timeout-s", type=int, default=1200)

    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-pdf", type=Path, default=None)
    p.add_argument("--skip-pdf", action="store_true")
    p.add_argument("--retain-stage-events", action="store_true")
    p.add_argument("--require-at-least-one-noisy", action="store_true", default=True)
    p.add_argument("--smoke-test-intentionally-weak", action="store_true", help="# SMOKE TEST - intentionally weak settings")

    return p.parse_args(argv)


def _validate_pool_b_strict_composition(pool_b_meta: dict[str, Any]) -> dict[str, Any]:
    required = {"uccsd", "hva", "paop_full"}
    raw = pool_b_meta.get("raw_sizes", {})
    if not isinstance(raw, dict):
        raise ValueError("Pool B metadata missing raw_sizes.")
    if set(raw.keys()) != required:
        raise ValueError(
            "Pool B composition mismatch: expected exactly raw_sizes keys "
            f"{sorted(required)}, got {sorted(raw.keys())}."
        )
    missing = [fam for fam in sorted(required) if int(raw.get(fam, 0)) <= 0]
    if missing:
        raise ValueError(
            "Pool B composition invalid: each family must be non-empty. "
            f"Missing/empty={missing}."
        )
    dedup_presence = pool_b_meta.get("dedup_source_presence_counts", {})
    if not isinstance(dedup_presence, dict):
        raise ValueError("Pool B metadata missing dedup_source_presence_counts.")
    if set(dedup_presence.keys()) != required:
        raise ValueError(
            "Pool B dedup presence mismatch: expected keys "
            f"{sorted(required)}, got {sorted(dedup_presence.keys())}."
        )
    return {
        "required_families": ["uccsd_lifted", "hva", "paop_full"],
        "raw_sizes": {k: int(raw[k]) for k in sorted(raw.keys())},
        "dedup_source_presence_counts": {
            k: int(dedup_presence[k]) for k in sorted(dedup_presence.keys())
        },
        "passed": True,
    }


def _collect_noisy_benchmark_rows(dynamics_noisy: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    profiles = dynamics_noisy.get("profiles", {}) if isinstance(dynamics_noisy, dict) else {}
    for profile_name, profile_data in profiles.items():
        methods = profile_data.get("methods", {}) if isinstance(profile_data, dict) else {}
        if not isinstance(methods, dict):
            continue
        for method_name, method_data in methods.items():
            modes = method_data.get("modes", {}) if isinstance(method_data, dict) else {}
            if not isinstance(modes, dict):
                continue
            for mode_name, mode_data in modes.items():
                if not bool(isinstance(mode_data, dict) and mode_data.get("success", False)):
                    continue
                cost = mode_data.get("benchmark_cost", {})
                runtime = mode_data.get("benchmark_runtime", {})
                delta_unc = mode_data.get("delta_uncertainty", {})
                energy_unc = {}
                if isinstance(delta_unc, dict):
                    energy_unc = delta_unc.get("energy_total", {}) if isinstance(delta_unc.get("energy_total", {}), dict) else {}
                rows.append(
                    {
                        "profile": str(profile_name),
                        "method": str(method_name),
                        "mode": str(mode_name),
                        "term_exp_count_total": int(cost.get("term_exp_count_total", 0)),
                        "pauli_rot_count_total": int(cost.get("pauli_rot_count_total", 0)),
                        "cx_proxy_total": int(cost.get("cx_proxy_total", 0)),
                        "sq_proxy_total": int(cost.get("sq_proxy_total", 0)),
                        "depth_proxy_total": int(cost.get("depth_proxy_total", 0)),
                        "wall_total_s": float(runtime.get("wall_total_s", float("nan"))),
                        "oracle_eval_s_total": float(runtime.get("oracle_eval_s_total", float("nan"))),
                        "oracle_calls_total": int(runtime.get("oracle_calls_total", 0)),
                        "max_abs_delta": float(energy_unc.get("max_abs_delta", float("nan"))),
                        "max_abs_delta_over_stderr": float(
                            energy_unc.get("max_abs_delta_over_stderr", float("nan"))
                        ),
                        "mean_abs_delta_over_stderr": float(
                            energy_unc.get("mean_abs_delta_over_stderr", float("nan"))
                        ),
                    }
                )
    return rows


def main(argv: list[str] | None = None) -> None:
    args = _enforce_defaults_and_minimums(parse_args(argv))
    noisy_methods = _parse_noisy_methods_csv(str(args.noisy_methods))
    mitigation_cfg = _build_mitigation_config(
        mitigation=str(args.mitigation),
        zne_scales=(None if args.zne_scales is None else str(args.zne_scales)),
        dd_sequence=(None if args.dd_sequence is None else str(args.dd_sequence)),
    )
    symmetry_mitigation_cfg = _build_symmetry_mitigation_config(
        mode=str(args.symmetry_mitigation_mode),
        L=int(args.L),
        ordering=str(args.ordering),
    )

    if int(args.L) != 2 and not bool(args.smoke_test_intentionally_weak):
        raise ValueError("This report is currently locked to L=2 for the full static+drive noise matrix.")

    transition_cfg = TransitionConfig(
        window_k=int(args.window_k),
        slope_epsilon=float(args.slope_epsilon),
        patience=int(args.patience),
        min_points_before_switch=int(args.min_points_before_switch),
    )

    artifacts_json_dir = REPO_ROOT / "artifacts" / "json"
    docs_dir = REPO_ROOT / "docs"
    artifacts_json_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    tag = (
        f"L{int(args.L)}_hh_u{float(args.u):g}_g{float(args.g_ep):g}_"
        f"S{int(args.trotter_steps)}_N{int(args.num_times)}"
    ).replace(".", "p")

    output_json = args.output_json or (artifacts_json_dir / f"hh_noise_robustness_L2_{tag}.json")
    output_pdf = args.output_pdf or (docs_dir / "HH noise robustness report.PDF")

    _ai_log("hh_noise_robustness_start", settings=vars(args))

    num_particles = _half_filled_particles(int(args.L))
    h_poly = _build_hh_hamiltonian(args)
    hmat = np.asarray(hamiltonian_matrix(h_poly), dtype=complex)
    native_order, coeff_map_exyz = _collect_hardcoded_terms_exyz(h_poly)
    ordered_labels_exyz = sorted(native_order)

    exact_sector_energy = float(
        exact_ground_energy_sector_hh(
            h_poly,
            num_sites=int(args.L),
            num_particles=(int(num_particles[0]), int(num_particles[1])),
            n_ph_max=int(args.n_ph_max),
            boson_encoding=str(args.boson_encoding),
            indexing=str(args.ordering),
        )
    )
    exact_basis_energy, psi_exact_ref, fidelity_basis_v0 = _compute_exact_reference_for_hh(
        hmat=np.asarray(hmat, dtype=complex),
        num_sites=int(args.L),
        ordering=str(args.ordering),
        num_particles=(int(num_particles[0]), int(num_particles[1])),
        energy_tol=float(args.fidelity_subspace_energy_tol),
    )
    exact_sector_energy = float(exact_basis_energy)

    psi_ref = _build_reference_state(args, num_particles)

    # Stage 1: warm-start HVA
    warm_ansatz = _build_hh_hva_ansatz(args, reps=int(args.warm_reps))
    warm_payload, psi_warm = _run_vqe_stage_with_transition(
        stage_name="hva_warm_start",
        h_poly=h_poly,
        ansatz=warm_ansatz,
        psi_ref=psi_ref,
        exact_energy=float(exact_sector_energy),
        restarts=int(args.warm_restarts),
        seed=int(args.warm_seed),
        maxiter=int(args.warm_maxiter),
        method=str(args.warm_method),
        transition_cfg=transition_cfg,
    )
    _ai_log(
        "hh_noise_robustness_stage_done",
        stage="warm_start",
        energy=float(warm_payload.get("energy", float("nan"))),
        delta_abs=float(warm_payload.get("delta_abs", float("nan"))),
        stop_reason=str(warm_payload.get("stop_reason", "")),
    )

    # Stage 2: ADAPT with strict Pool B union
    uccsd_lifted = _build_uccsd_fermion_lifted_pool(
        num_sites=int(args.L),
        num_particles=(int(num_particles[0]), int(num_particles[1])),
        n_ph_max=int(args.n_ph_max),
        boson_encoding=str(args.boson_encoding),
        ordering=str(args.ordering),
    )
    hva_pool = _build_hva_pool(args)
    paop_full_pool = _build_paop_full_pool(args=args, num_particles=num_particles)

    pool_B, pool_B_meta, pool_B_source_by_sig = build_pool_b_strict_union(
        uccsd_ops=uccsd_lifted,
        hva_ops=hva_pool,
        paop_full_ops=paop_full_pool,
    )
    pool_b_audit = _validate_pool_b_strict_composition(pool_B_meta)

    adapt_payload, psi_adapt = _run_adapt_stage_with_transition(
        h_poly=h_poly,
        psi_start=np.asarray(psi_warm, dtype=complex),
        pool=pool_B,
        exact_energy=float(exact_sector_energy),
        allow_repeats=bool(args.adapt_allow_repeats),
        max_depth=int(args.adapt_max_depth),
        maxiter=int(args.adapt_maxiter),
        eps_grad=float(args.adapt_eps_grad),
        eps_energy=float(args.adapt_eps_energy),
        seed=int(args.adapt_seed),
        transition_cfg=transition_cfg,
    )
    _ai_log(
        "hh_noise_robustness_stage_done",
        stage="adapt_pool_b",
        energy=float(adapt_payload.get("energy", float("nan"))),
        delta_abs=float(adapt_payload.get("delta_abs", float("nan"))),
        stop_reason=str(adapt_payload.get("stop_reason", "")),
    )

    # Stage 3: conventional VQE seeded from ADAPT state
    final_ansatz = _build_hh_hva_ansatz(args, reps=int(args.final_reps))
    final_payload, psi_final = _run_vqe_stage_with_transition(
        stage_name="conventional_vqe_seeded_from_adapt",
        h_poly=h_poly,
        ansatz=final_ansatz,
        psi_ref=np.asarray(psi_adapt, dtype=complex),
        exact_energy=float(exact_sector_energy),
        restarts=int(args.final_restarts),
        seed=int(args.final_seed),
        maxiter=int(args.final_maxiter),
        method=str(args.final_method),
        transition_cfg=transition_cfg,
    )
    _ai_log(
        "hh_noise_robustness_stage_done",
        stage="conventional_vqe",
        energy=float(final_payload.get("energy", float("nan"))),
        delta_abs=float(final_payload.get("delta_abs", float("nan"))),
        stop_reason=str(final_payload.get("stop_reason", "")),
    )

    if not bool(args.retain_stage_events):
        warm_payload.pop("progress_events", None)

    disable_time_dynamics = bool(args.disable_time_dynamics)
    # Noiseless/noisy profiles use static + optional drive semantics.
    static_profile = None
    drive_profile = _build_drive_profile(args, enabled=bool(args.include_drive_profile))

    hardcoded_superset: dict[str, Any] = _disabled_hardcoded_superset_meta()
    noise_modes = [x.strip() for x in str(args.noise_modes).split(",") if x.strip()]
    noisy_mode_diagnostics: dict[str, Any] = {}
    noisy_audit_diagnostics: dict[str, Any] = {}
    noisy_final_audit: dict[str, Any] = {"enabled": bool(disable_time_dynamics), "profiles": {}}

    if not disable_time_dynamics:
        dynamics_noiseless = {
            "profiles": {
                "static": _run_noiseless_profile(
                    args=args,
                    psi_seed=np.asarray(psi_final, dtype=complex),
                    hmat=hmat,
                    ordered_labels_exyz=ordered_labels_exyz,
                    coeff_map_exyz=coeff_map_exyz,
                    drive_profile=static_profile,
                )
            }
        }
        if drive_profile is not None:
            dynamics_noiseless["profiles"]["drive"] = _run_noiseless_profile(
                args=args,
                psi_seed=np.asarray(psi_final, dtype=complex),
                hmat=hmat,
                ordered_labels_exyz=ordered_labels_exyz,
                coeff_map_exyz=coeff_map_exyz,
                drive_profile=drive_profile,
            )

        dynamics_noisy = {"profiles": {}}
        for profile_name, profile_cfg in [("static", static_profile), ("drive", drive_profile)]:
            if profile_name == "drive" and profile_cfg is None:
                continue

            method_payloads: dict[str, Any] = {}
            for method in noisy_methods:
                profile_modes: dict[str, Any] = {}
                for mode in noise_modes:
                    kwargs = {
                        "L": int(args.L),
                        "ordering": str(args.ordering),
                        "psi_seed": np.asarray(psi_final, dtype=complex),
                        "ordered_labels_exyz": list(ordered_labels_exyz),
                        "static_coeff_map_exyz": dict(coeff_map_exyz),
                        "t_final": float(args.t_final),
                        "num_times": int(args.num_times),
                        "trotter_steps": int(args.trotter_steps),
                        "drive_profile": profile_cfg,
                        "noise_mode": str(mode),
                        "shots": int(args.shots),
                        "seed": int(args.noise_seed),
                        "oracle_repeats": int(args.oracle_repeats),
                        "oracle_aggregate": str(args.oracle_aggregate),
                        "mitigation_config": dict(mitigation_cfg),
                        "symmetry_mitigation_config": dict(symmetry_mitigation_cfg),
                        "backend_name": (None if args.backend_name is None else str(args.backend_name)),
                        "use_fake_backend": bool(args.use_fake_backend),
                        "allow_aer_fallback": bool(args.allow_aer_fallback),
                        "omp_shm_workaround": bool(args.omp_shm_workaround),
                        "method": str(method),
                        "benchmark_active_coeff_tol": float(args.benchmark_active_coeff_tol),
                        "cfqm_coeff_drop_abs_tol": float(args.cfqm_coeff_drop_abs_tol),
                    }
                    mode_result = _run_noisy_mode_isolated(
                        kwargs=kwargs,
                        timeout_s=int(args.noisy_mode_timeout_s),
                    )
                    profile_modes[str(mode)] = mode_result
                    noisy_mode_diagnostics[f"{profile_name}:{method}:{mode}"] = {
                        "success": bool(mode_result.get("success", False)),
                        "env_blocked": bool(mode_result.get("env_blocked", False)),
                        "reason": mode_result.get("reason", None),
                        "error": mode_result.get("error", None),
                    }
                method_payloads[str(method)] = {"modes": profile_modes}

            suzuki_alias_modes = (
                method_payloads.get("suzuki2", {}).get("modes", {})
                if isinstance(method_payloads.get("suzuki2", {}), dict)
                else {}
            )
            dynamics_noisy["profiles"][profile_name] = {
                "drive_enabled": bool(profile_cfg is not None),
                "methods": method_payloads,
                "modes": suzuki_alias_modes,
            }
        dynamics_benchmark_rows = _collect_noisy_benchmark_rows(dynamics_noisy)
    else:
        dynamics_noiseless = {
            "profiles": {},
            "disabled": True,
            "reason": "time dynamics disabled by --disable-time-dynamics",
        }
        dynamics_noisy = {
            "profiles": {},
            "disabled": True,
            "reason": "time dynamics disabled by --disable-time-dynamics",
        }
        dynamics_benchmark_rows = []
        for profile_name, profile_cfg in [("static", static_profile), ("drive", drive_profile)]:
            if profile_name == "drive" and profile_cfg is None:
                continue
            profile_modes: dict[str, Any] = {}
            for mode in noise_modes:
                kwargs = {
                    "L": int(args.L),
                    "ordering": str(args.ordering),
                    "psi_seed": np.asarray(psi_final, dtype=complex),
                    "ordered_labels_exyz": list(ordered_labels_exyz),
                    "static_coeff_map_exyz": dict(coeff_map_exyz),
                    "drive_profile": profile_cfg,
                    "noise_mode": str(mode),
                    "shots": int(args.shots),
                    "seed": int(args.noise_seed),
                    "oracle_repeats": int(args.oracle_repeats),
                    "oracle_aggregate": str(args.oracle_aggregate),
                    "mitigation_config": dict(mitigation_cfg),
                    "symmetry_mitigation_config": dict(symmetry_mitigation_cfg),
                    "backend_name": (None if args.backend_name is None else str(args.backend_name)),
                    "use_fake_backend": bool(args.use_fake_backend),
                    "allow_aer_fallback": bool(args.allow_aer_fallback),
                    "omp_shm_workaround": bool(args.omp_shm_workaround),
                }
                mode_result = _run_noisy_audit_mode_isolated(
                    kwargs=kwargs,
                    timeout_s=int(args.noisy_mode_timeout_s),
                )
                profile_modes[str(mode)] = mode_result
                noisy_audit_diagnostics[f"{profile_name}:{mode}"] = {
                    "success": bool(mode_result.get("success", False)),
                    "env_blocked": bool(mode_result.get("env_blocked", False)),
                    "reason": mode_result.get("reason", None),
                    "error": mode_result.get("error", None),
                }
            noisy_final_audit["profiles"][profile_name] = {
                "drive_enabled": bool(profile_cfg is not None),
                "modes": profile_modes,
            }

    run_command = current_command_string()

    payload: dict[str, Any] = {
        "settings": {
            "L": int(args.L),
            "problem": "hh",
            "t": float(args.t),
            "u": float(args.u),
            "dv": float(args.dv),
            "omega0": float(args.omega0),
            "g_ep": float(args.g_ep),
            "n_ph_max": int(args.n_ph_max),
            "boson_encoding": str(args.boson_encoding),
            "ordering": str(args.ordering),
            "boundary": str(args.boundary),
            "t_final": float(args.t_final),
            "num_times": int(args.num_times),
            "trotter_steps": int(args.trotter_steps),
            "exact_steps_multiplier": int(args.exact_steps_multiplier),
            "disable_time_dynamics": bool(disable_time_dynamics),
            "fidelity_subspace_energy_tol": float(args.fidelity_subspace_energy_tol),
            "include_drive_profile": bool(args.include_drive_profile),
            "smoke_test_intentionally_weak": bool(args.smoke_test_intentionally_weak),
            "noise_modes": noise_modes,
            "noisy_methods": [str(x) for x in noisy_methods],
            "shots": int(args.shots),
            "oracle_repeats": int(args.oracle_repeats),
            "oracle_aggregate": str(args.oracle_aggregate),
            "mitigation": str(args.mitigation),
            "symmetry_mitigation_mode": str(args.symmetry_mitigation_mode),
            "zne_scales": (None if args.zne_scales is None else str(args.zne_scales)),
            "dd_sequence": (None if args.dd_sequence is None else str(args.dd_sequence)),
            "mitigation_config": dict(mitigation_cfg),
            "symmetry_mitigation_config": dict(symmetry_mitigation_cfg),
            "benchmark_active_coeff_tol": float(args.benchmark_active_coeff_tol),
            "drive_profile": drive_profile,
            "transition_policy": {
                "window_k": int(args.window_k),
                "slope_epsilon": float(args.slope_epsilon),
                "patience": int(args.patience),
                "min_points_before_switch": int(args.min_points_before_switch),
            },
        },
        "stage_pipeline": {
            "warm_start": warm_payload,
            "adapt_pool_b": adapt_payload,
            "conventional_vqe": final_payload,
        },
        "transitions": {
            "warm_to_adapt": dict(warm_payload.get("transition", {})),
            "adapt_to_vqe": dict(adapt_payload.get("transition", {})),
        },
        "pool_B": {
            **pool_B_meta,
            "strict_union_families": ["uccsd_lifted", "hva", "paop_full"],
        },
        "hardcoded_superset": hardcoded_superset,
        "dynamics_noiseless": dynamics_noiseless,
        "dynamics_noisy": dynamics_noisy,
        "noisy_final_audit": noisy_final_audit,
        "dynamics_benchmarks": {
            "rows": dynamics_benchmark_rows,
            "metric_fields": [
                "term_exp_count_total",
                "pauli_rot_count_total",
                "cx_proxy_total",
                "sq_proxy_total",
                "depth_proxy_total",
                "wall_total_s",
                "oracle_eval_s_total",
                "oracle_calls_total",
                "max_abs_delta",
                "max_abs_delta_over_stderr",
                "mean_abs_delta_over_stderr",
            ],
        },
        "comparisons": {},
        "summary": {},
        "equation_registry": {},
        "plot_contracts": {},
        "diagnostics": {
            "noisy_mode_diagnostics": noisy_mode_diagnostics,
            "noisy_audit_diagnostics": noisy_audit_diagnostics,
            "pool_signature_space": int(len(pool_B_source_by_sig)),
            "pool_b_audit": pool_b_audit,
            "coeff_map_exyz": flatten_coeff_map_real_imag(coeff_map_exyz),
            "ordered_labels_count": int(len(ordered_labels_exyz)),
            "exact_sector_energy": float(exact_sector_energy),
            "exact_basis_energy": float(exact_basis_energy),
            "metric_definitions": {
                "delta_abs": "abs(E - E_exact_sector)",
                "energy_static": "<psi|H_static|psi>",
                "energy_total": "<psi|H_static + H_drive(t)|psi>",
                "doublon": "<n_up(site0) n_dn(site0)>",
                "staggered": "(1/L) sum_j (-1)^j <n_j>",
            },
        },
        "run_command": str(run_command),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    payload["comparisons"] = _compute_comparisons(payload)
    payload["summary"] = _build_summary(payload)
    equation_registry, plot_contracts = _build_equation_registry_and_contracts(payload)
    payload["equation_registry"] = equation_registry
    payload["plot_contracts"] = plot_contracts

    _json_dump(output_json, payload)

    if not bool(args.skip_pdf):
        _write_pdf(output_pdf, payload)

    if bool(args.require_at_least_one_noisy):
        completed = int(
            payload["summary"].get(
                "noisy_method_modes_completed",
                payload["summary"].get("noisy_modes_completed", 0),
            )
        )
        if completed < 1:
            completed = int(payload["summary"].get("noisy_audit_modes_completed", 0))
        if completed < 1:
            raise RuntimeError(
                "No noisy mode completed successfully. "
                "Set --require-at-least-one-noisy false to treat this as non-fatal."
            )

    _ai_log(
        "hh_noise_robustness_done",
        output_json=str(output_json),
        output_pdf=(None if bool(args.skip_pdf) else str(output_pdf)),
        noisy_completed=int(payload["summary"].get("noisy_method_modes_completed", payload["summary"].get("noisy_modes_completed", 0))),
        noisy_total=int(payload["summary"].get("noisy_method_modes_total", payload["summary"].get("noisy_modes_total", 0))),
        noisy_audit_completed=int(payload["summary"].get("noisy_audit_modes_completed", 0)),
        noisy_audit_total=int(payload["summary"].get("noisy_audit_modes_total", 0)),
    )


if __name__ == "__main__":
    main()
