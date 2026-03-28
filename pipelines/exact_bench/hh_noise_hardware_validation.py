#!/usr/bin/env python3
"""HH-first noise/hardware validation runner (wrapper-level).

This script validates feasibility of HH/Hubbard workflows under measurement and
device noise without mutating core operator algebra files.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
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
from docs.reports.report_pages import (
    render_executive_summary_page,
    render_manifest_overview_page,
    render_section_divider_page,
)
from src.quantum.compiled_polynomial import (
    compile_polynomial_action,
    energy_via_one_apply,
)
from src.quantum.hartree_fock_reference_state import (
    hartree_fock_statevector,
    hubbard_holstein_reference_state,
)
from src.quantum.spsa_optimizer import spsa_minimize
from src.quantum.hubbard_latex_python_pairs import (
    build_hubbard_hamiltonian,
    build_hubbard_holstein_hamiltonian,
)
from src.quantum.vqe_latex_python_pairs import (
    AnsatzTerm,
    HardcodedUCCSDAnsatz,
    HubbardHolsteinLayerwiseAnsatz,
    HubbardHolsteinPhysicalTermwiseAnsatz,
    HubbardHolsteinTermwiseAnsatz,
    HubbardLayerwiseAnsatz,
    exact_ground_energy_sector,
    exact_ground_energy_sector_hh,
    hamiltonian_matrix,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm

from pipelines.exact_bench.noise_oracle_runtime import (
    ExpectationOracle,
    NoiseBackendInfo,
    OracleConfig,
    _ansatz_to_circuit,
    _doublon_site_qop,
    _number_operator_qop,
    _ordered_qop_from_exyz,
    _pauli_poly_to_sparse_pauli_op,
    normalize_ideal_reference_symmetry_mitigation,
    normalize_mitigation_config,
    normalize_symmetry_mitigation_config,
)
from pipelines.hardcoded.adapt_pipeline import _oracle_fd_gradient_stderr


def _ai_log(event: str, **fields: Any) -> None:
    payload = {"event": str(event), "ts_utc": datetime.now(timezone.utc).isoformat(), **fields}
    print(f"AI_LOG {json.dumps(payload, sort_keys=True, default=str)}", flush=True)


_HUBBARD_PARAMS: dict[int, dict[str, Any]] = {
    2: {"trotter_steps": 64, "exact_mult": 2, "num_times": 201, "reps": 2, "restarts": 2, "maxiter": 600, "method": "COBYLA", "t_final": 10.0},
    3: {"trotter_steps": 128, "exact_mult": 2, "num_times": 201, "reps": 2, "restarts": 3, "maxiter": 1200, "method": "COBYLA", "t_final": 15.0},
    4: {"trotter_steps": 256, "exact_mult": 3, "num_times": 241, "reps": 3, "restarts": 4, "maxiter": 6000, "method": "SLSQP", "t_final": 20.0},
    5: {"trotter_steps": 384, "exact_mult": 3, "num_times": 301, "reps": 3, "restarts": 5, "maxiter": 8000, "method": "SLSQP", "t_final": 20.0},
    6: {"trotter_steps": 512, "exact_mult": 4, "num_times": 361, "reps": 4, "restarts": 6, "maxiter": 10000, "method": "SLSQP", "t_final": 20.0},
}

_HH_PARAMS: dict[tuple[int, int], dict[str, Any]] = {
    (2, 1): {"trotter_steps": 64, "reps": 2, "restarts": 3, "maxiter": 800, "method": "COBYLA"},
    (2, 2): {"trotter_steps": 128, "reps": 3, "restarts": 4, "maxiter": 1500, "method": "COBYLA"},
    (3, 1): {"trotter_steps": 192, "reps": 2, "restarts": 4, "maxiter": 2400, "method": "COBYLA"},
}


def _half_filled_particles(num_sites: int) -> tuple[int, int]:
    return ((int(num_sites) + 1) // 2, int(num_sites) // 2)


def _fermion_mode_index(site: int, spin: str, ordering: str, num_sites: int) -> int:
    s = str(spin).strip().lower()
    ord_norm = str(ordering).strip().lower()
    if ord_norm == "blocked":
        if s in {"up", "u", "alpha"}:
            return int(site)
        return int(num_sites) + int(site)
    if s in {"up", "u", "alpha"}:
        return 2 * int(site)
    return 2 * int(site) + 1


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
    max_z = float(np.max(finite_z)) if finite_z.size > 0 else float("nan")
    mean_z = float(np.mean(finite_z)) if finite_z.size > 0 else float("nan")
    return {
        "max_abs_delta": float(np.max(delta_abs)) if delta_abs.size > 0 else float("nan"),
        "max_abs_delta_over_stderr": max_z,
        "mean_abs_delta_over_stderr": mean_z,
    }


def _build_mitigation_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return normalize_mitigation_config(
        {
            "mode": str(args.mitigation),
            "zne_scales": args.zne_scales,
            "dd_sequence": args.dd_sequence,
        }
    )


def _build_symmetry_mitigation_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    n_up, n_dn = _half_filled_particles(int(args.L))
    return normalize_symmetry_mitigation_config(
        {
            "mode": str(args.symmetry_mitigation_mode),
            "num_sites": int(args.L),
            "ordering": str(args.ordering),
            "sector_n_up": int(n_up),
            "sector_n_dn": int(n_dn),
        }
    )


def _mitigation_caption(
    cfg: dict[str, Any] | None,
    symmetry_cfg: dict[str, Any] | None = None,
) -> str:
    data = cfg or {}
    sym = symmetry_cfg or {}
    return (
        f"mitigation={data.get('mode', 'none')}, "
        f"zne_scales={data.get('zne_scales', [])}, "
        f"dd_sequence={data.get('dd_sequence', None)}, "
        f"symmetry={sym.get('mode', 'off')}"
    )


_SUZUKI2_MATH = "U(t) ~= [prod_j exp(-i (t/r)/2 H_j) * prod_j^rev exp(-i (t/r)/2 H_j)]^r"


def _trotterized_circuit(
    initial_circuit: QuantumCircuit,
    ordered_qop: SparsePauliOp,
    *,
    time_value: float,
    trotter_steps: int,
    suzuki_order: int,
) -> QuantumCircuit:
    if int(suzuki_order) != 2:
        raise ValueError("This validation runner currently supports suzuki_order=2 only.")
    qc = initial_circuit.copy()
    evo = PauliEvolutionGate(
        ordered_qop,
        time=float(time_value),
        synthesis=SuzukiTrotter(order=int(suzuki_order), reps=int(trotter_steps), preserve_order=True),
    )
    qc.append(evo, list(range(int(initial_circuit.num_qubits))))
    return qc


def _get_hubbard_minimum(L: int) -> dict[str, Any]:
    if int(L) in _HUBBARD_PARAMS:
        return dict(_HUBBARD_PARAMS[int(L)])
    base = dict(_HUBBARD_PARAMS[max(_HUBBARD_PARAMS)])
    scale = float(L) / float(max(_HUBBARD_PARAMS))
    base["trotter_steps"] = int(round(base["trotter_steps"] * scale))
    base["maxiter"] = int(round(base["maxiter"] * scale))
    base["reps"] = max(base["reps"], int(round(base["reps"] * scale)))
    base["restarts"] = max(base["restarts"], int(round(base["restarts"] * scale)))
    return base


def _get_hh_minimum(L: int, n_ph_max: int) -> dict[str, Any]:
    key = (int(L), int(n_ph_max))
    if key in _HH_PARAMS:
        return dict(_HH_PARAMS[key])
    fallback = _get_hubbard_minimum(int(L))
    return {
        "trotter_steps": int(round(float(fallback["trotter_steps"]) * 1.5)),
        "reps": int(fallback["reps"]),
        "restarts": int(fallback["restarts"]) + 1,
        "maxiter": int(round(float(fallback["maxiter"]) * 1.5)),
        "method": str(fallback["method"]),
    }


def _apply_defaults_and_minimums(args: argparse.Namespace) -> argparse.Namespace:
    problem = str(args.problem).strip().lower()
    if problem == "hh":
        minimum = _get_hh_minimum(int(args.L), int(args.n_ph_max))
    else:
        minimum = _get_hubbard_minimum(int(args.L))

    if args.vqe_reps is None:
        args.vqe_reps = int(minimum["reps"])
    if args.vqe_restarts is None:
        args.vqe_restarts = int(minimum["restarts"])
    if args.vqe_maxiter is None:
        args.vqe_maxiter = int(minimum["maxiter"])
    if args.trotter_steps is None:
        args.trotter_steps = int(minimum["trotter_steps"])
    if args.vqe_method is None:
        args.vqe_method = str(minimum["method"])

    if problem == "hubbard":
        if args.t_final is None:
            args.t_final = float(minimum["t_final"])
        if args.num_times is None:
            args.num_times = int(minimum["num_times"])
        if args.exact_steps_multiplier is None:
            args.exact_steps_multiplier = int(minimum["exact_mult"])
    else:
        if args.t_final is None:
            args.t_final = 20.0
        if args.num_times is None:
            args.num_times = 201
        if args.exact_steps_multiplier is None:
            args.exact_steps_multiplier = 1

    legacy_parity_mode = getattr(args, "legacy_reference_json", None) is not None
    if (not bool(args.smoke_test_intentionally_weak)) and (not legacy_parity_mode):
        checks = {
            "vqe_reps": int(args.vqe_reps) >= int(minimum["reps"]),
            "vqe_restarts": int(args.vqe_restarts) >= int(minimum["restarts"]),
            "vqe_maxiter": int(args.vqe_maxiter) >= int(minimum["maxiter"]),
            "trotter_steps": int(args.trotter_steps) >= int(minimum["trotter_steps"]),
        }
        failed = [k for k, ok in checks.items() if not ok]
        if failed:
            raise ValueError(
                "Under-parameterized run rejected by AGENTS minimum table. "
                f"Failed fields: {failed}. "
                f"Minimums used for this case: {minimum}. "
                "Pass --smoke-test-intentionally-weak only for explicit smoke tests."
            )
    return args


def _build_hamiltonian(args: argparse.Namespace) -> Any:
    if str(args.problem).strip().lower() == "hh":
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
    return build_hubbard_hamiltonian(
        dims=int(args.L),
        t=float(args.t),
        U=float(args.u),
        v=float(args.dv),
        repr_mode="JW",
        indexing=str(args.ordering),
        pbc=(str(args.boundary).strip().lower() == "periodic"),
    )


def _build_ansatz(args: argparse.Namespace, num_particles: tuple[int, int]) -> Any:
    problem = str(args.problem).strip().lower()
    ans = str(args.ansatz).strip().lower()
    pbc = (str(args.boundary).strip().lower() == "periodic")

    if ans == "hh_hva":
        if problem != "hh":
            raise ValueError("ansatz=hh_hva requires --problem hh")
        return HubbardHolsteinLayerwiseAnsatz(
            dims=int(args.L),
            J=float(args.t),
            U=float(args.u),
            omega0=float(args.omega0),
            g=float(args.g_ep),
            n_ph_max=int(args.n_ph_max),
            boson_encoding=str(args.boson_encoding),
            reps=int(args.vqe_reps),
            repr_mode="JW",
            indexing=str(args.ordering),
            pbc=pbc,
        )
    if ans == "hh_hva_tw":
        if problem != "hh":
            raise ValueError("ansatz=hh_hva_tw requires --problem hh")
        return HubbardHolsteinTermwiseAnsatz(
            dims=int(args.L),
            J=float(args.t),
            U=float(args.u),
            omega0=float(args.omega0),
            g=float(args.g_ep),
            n_ph_max=int(args.n_ph_max),
            boson_encoding=str(args.boson_encoding),
            reps=int(args.vqe_reps),
            repr_mode="JW",
            indexing=str(args.ordering),
            pbc=pbc,
        )
    if ans == "hh_hva_ptw":
        if problem != "hh":
            raise ValueError("ansatz=hh_hva_ptw requires --problem hh")
        return HubbardHolsteinPhysicalTermwiseAnsatz(
            dims=int(args.L),
            J=float(args.t),
            U=float(args.u),
            omega0=float(args.omega0),
            g=float(args.g_ep),
            n_ph_max=int(args.n_ph_max),
            boson_encoding=str(args.boson_encoding),
            reps=int(args.vqe_reps),
            repr_mode="JW",
            indexing=str(args.ordering),
            pbc=pbc,
        )
    if ans == "hva":
        if problem != "hubbard":
            raise ValueError("ansatz=hva currently requires --problem hubbard")
        return HubbardLayerwiseAnsatz(
            dims=int(args.L),
            t=float(args.t),
            U=float(args.u),
            v=float(args.dv),
            reps=int(args.vqe_reps),
            repr_mode="JW",
            indexing=str(args.ordering),
            pbc=pbc,
            include_potential_terms=True,
        )
    if ans == "uccsd":
        if problem != "hubbard":
            raise ValueError("ansatz=uccsd currently requires --problem hubbard")
        return HardcodedUCCSDAnsatz(
            dims=int(args.L),
            num_particles=num_particles,
            reps=int(args.vqe_reps),
            repr_mode="JW",
            indexing=str(args.ordering),
            include_singles=True,
            include_doubles=True,
        )

    raise ValueError(f"Unsupported ansatz '{ans}'")


def _build_reference_state(
    *,
    args: argparse.Namespace,
    num_particles: tuple[int, int],
) -> np.ndarray:
    if str(args.problem).strip().lower() == "hh":
        return np.asarray(
            hubbard_holstein_reference_state(
                dims=int(args.L),
                num_particles=num_particles,
                n_ph_max=int(args.n_ph_max),
                boson_encoding=str(args.boson_encoding),
                indexing=str(args.ordering),
            ),
            dtype=complex,
        ).reshape(-1)
    return np.asarray(
        hartree_fock_statevector(
            int(args.L),
            num_particles,
            indexing=str(args.ordering),
        ),
        dtype=complex,
    ).reshape(-1)


def _build_adapt_pool(
    *,
    args: argparse.Namespace,
    h_poly: Any,
    num_particles: tuple[int, int],
) -> list[AnsatzTerm]:
    problem = str(args.problem).strip().lower()
    pool_key = str(args.adapt_pool).strip().lower()

    if problem == "hubbard":
        if pool_key == "uccsd":
            return list(
                HardcodedUCCSDAnsatz(
                    dims=int(args.L),
                    num_particles=num_particles,
                    reps=1,
                    repr_mode="JW",
                    indexing=str(args.ordering),
                    include_singles=True,
                    include_doubles=True,
                ).base_terms
            )
        if pool_key == "full_hamiltonian":
            out: list[AnsatzTerm] = []
            for term in h_poly.return_polynomial():
                coeff = complex(term.p_coeff)
                if abs(coeff) <= 1e-15:
                    continue
                lbl = str(term.pw2strng())
                nq = int(term.nqubit())
                if lbl == ("e" * nq):
                    continue
                p = PauliPolynomial(nq)
                p.add_term(PauliTerm(nq, ps=lbl, pc=1.0))
                out.append(AnsatzTerm(label=f"ham_term({lbl})", polynomial=p))
            return out
        raise ValueError(
            f"Unsupported Hubbard ADAPT pool '{pool_key}'. Use uccsd or full_hamiltonian."
        )

    # HH
    if pool_key == "hva":
        # Phase-2 default: use HH physical-termwise generators as pool entries.
        return list(
            HubbardHolsteinPhysicalTermwiseAnsatz(
                dims=int(args.L),
                J=float(args.t),
                U=float(args.u),
                omega0=float(args.omega0),
                g=float(args.g_ep),
                n_ph_max=int(args.n_ph_max),
                boson_encoding=str(args.boson_encoding),
                reps=1,
                repr_mode="JW",
                indexing=str(args.ordering),
                pbc=(str(args.boundary).strip().lower() == "periodic"),
            ).base_terms
        )
    if pool_key == "full_hamiltonian":
        out = []
        for term in h_poly.return_polynomial():
            coeff = complex(term.p_coeff)
            if abs(coeff) <= 1e-15:
                continue
            lbl = str(term.pw2strng())
            nq = int(term.nqubit())
            if lbl == ("e" * nq):
                continue
            p = PauliPolynomial(nq)
            p.add_term(PauliTerm(nq, ps=lbl, pc=1.0))
            out.append(AnsatzTerm(label=f"ham_term({lbl})", polynomial=p))
        return out

    raise ValueError(
        f"Unsupported HH ADAPT pool '{pool_key}'. Use hva or full_hamiltonian."
    )


def _adapt_ops_to_circuit(
    ops: list[AnsatzTerm],
    theta: np.ndarray,
    *,
    num_qubits: int,
    reference_state: np.ndarray,
) -> QuantumCircuit:
    if int(theta.size) != int(len(ops)):
        raise ValueError(
            f"theta length {int(theta.size)} does not match selected ADAPT ops {int(len(ops))}"
        )
    qc = QuantumCircuit(int(num_qubits))
    from pipelines.exact_bench.noise_oracle_runtime import _append_reference_state  # local import

    _append_reference_state(qc, np.asarray(reference_state, dtype=complex))
    synthesis = SuzukiTrotter(order=2, reps=1, preserve_order=True)
    for op, ang in zip(ops, np.asarray(theta, dtype=float)):
        qop = _pauli_poly_to_sparse_pauli_op(op.polynomial)
        if np.max(np.abs(np.asarray(qop.coeffs, dtype=complex))) <= 1e-12:
            continue
        qc.append(PauliEvolutionGate(qop, time=float(ang), synthesis=synthesis), list(range(int(num_qubits))))
    return qc


def _run_noisy_adapt(
    *,
    args: argparse.Namespace,
    pool: list[AnsatzTerm],
    psi_ref: np.ndarray,
    h_qop: SparsePauliOp,
    noisy_oracle: ExpectationOracle,
    ideal_oracle: ExpectationOracle,
) -> tuple[dict[str, Any], list[AnsatzTerm], np.ndarray]:
    t0 = time.perf_counter()
    if not pool:
        raise ValueError("ADAPT pool is empty.")

    adapt_inner_key = str(args.adapt_inner_optimizer).strip().upper()
    adapt_spsa_eval_agg = str(args.adapt_spsa_eval_agg).strip().lower()
    adapt_spsa_params = {
        "a": float(args.adapt_spsa_a),
        "c": float(args.adapt_spsa_c),
        "alpha": float(args.adapt_spsa_alpha),
        "gamma": float(args.adapt_spsa_gamma),
        "A": float(args.adapt_spsa_A),
        "avg_last": int(args.adapt_spsa_avg_last),
        "eval_repeats": int(args.adapt_spsa_eval_repeats),
        "eval_agg": str(adapt_spsa_eval_agg),
    }

    minimize = None
    if adapt_inner_key != "SPSA":
        try:
            from scipy.optimize import minimize as _scipy_minimize
        except Exception as exc:
            raise RuntimeError("SciPy is required for phase-2 ADAPT COBYLA optimization.") from exc
        minimize = _scipy_minimize

    rng = np.random.default_rng(int(args.adapt_seed))
    max_depth = int(args.adapt_max_depth)
    grad_step = float(args.adapt_gradient_step)
    eps_grad = float(args.adapt_eps_grad)
    eps_energy = float(args.adapt_eps_energy)
    min_conf = float(args.adapt_min_confidence)
    allow_repeats = bool(args.adapt_allow_repeats)

    selected_ops: list[AnsatzTerm] = []
    theta = np.zeros(0, dtype=float)
    available = set(range(len(pool)))
    history: list[dict[str, Any]] = []
    nfev_total = 0
    stop_reason = "max_depth"
    phase1_score_z_alpha_used = 0.0
    gradient_uncertainty_source = "oracle_fd_stderr_v1"
    selection_metric_name = "g_abs"
    gradient_confidence_mode_used = "std"
    last_candidate_gradient_scout: list[dict[str, Any]] = []
    last_sigma_hat = 0.0
    last_g_lcb = 0.0
    last_gradient_confidence = 0.0
    last_gradient_confidence_stderr = 0.0
    last_selected_pool_index: int | None = None
    last_selected_label: str | None = None

    qc0 = _adapt_ops_to_circuit(
        selected_ops,
        theta,
        num_qubits=int(h_qop.num_qubits),
        reference_state=psi_ref,
    )
    e0 = noisy_oracle.evaluate(qc0, h_qop)
    energy_current = float(e0.mean)
    nfev_total += 1

    for depth in range(max_depth):
        candidate_indices = list(range(len(pool))) if allow_repeats else sorted(available)
        if not candidate_indices:
            stop_reason = "pool_exhausted"
            break

        best_idx = None
        best_abs_grad = -1.0
        best_grad = 0.0
        best_grad_std = 0.0
        best_grad_stderr = 0.0
        candidate_gradient_scout: list[dict[str, Any]] = []

        for idx in candidate_indices:
            trial_ops = selected_ops + [pool[idx]]
            t_plus = np.append(theta, grad_step)
            t_minus = np.append(theta, -grad_step)
            c_plus = _adapt_ops_to_circuit(
                trial_ops, t_plus, num_qubits=int(h_qop.num_qubits), reference_state=psi_ref
            )
            c_minus = _adapt_ops_to_circuit(
                trial_ops, t_minus, num_qubits=int(h_qop.num_qubits), reference_state=psi_ref
            )
            e_plus = noisy_oracle.evaluate(c_plus, h_qop)
            e_minus = noisy_oracle.evaluate(c_minus, h_qop)
            nfev_total += 2

            grad = float((e_plus.mean - e_minus.mean) / (2.0 * grad_step))
            grad_std = float(np.sqrt(e_plus.std ** 2 + e_minus.std ** 2) / (2.0 * grad_step))
            try:
                grad_stderr = float(_oracle_fd_gradient_stderr(e_plus, e_minus, grad_step=grad_step))
            except Exception as exc:
                raise RuntimeError(
                    "Failed to resolve oracle finite-difference gradient stderr "
                    f"at depth {int(depth + 1)} for candidate index {int(idx)} "
                    f"label '{str(pool[int(idx)].label)}'."
                ) from exc
            abs_grad = abs(grad)
            g_lcb = max(float(abs_grad) - float(phase1_score_z_alpha_used) * float(grad_stderr), 0.0)
            candidate_gradient_scout.append(
                {
                    "candidate_pool_index": int(idx),
                    "candidate_label": str(pool[int(idx)].label),
                    "gradient_signed": float(grad),
                    "gradient_abs": float(abs_grad),
                    "gradient_std": float(grad_std),
                    "gradient_stderr": float(grad_stderr),
                    "sigma_hat": float(grad_stderr),
                    "g_lcb": float(g_lcb),
                    "selection_metric_value": float(abs_grad),
                    "oracle_samples_plus": int(getattr(e_plus, "n_samples", 0)),
                    "oracle_samples_minus": int(getattr(e_minus, "n_samples", 0)),
                    "oracle_aggregate": str(getattr(e_plus, "aggregate", "mean")),
                    "selected_for_optimization": False,
                }
            )
            if abs_grad > best_abs_grad:
                best_abs_grad = abs_grad
                best_idx = int(idx)
                best_grad = float(grad)
                best_grad_std = float(grad_std)
                best_grad_stderr = float(grad_stderr)

        if best_idx is None:
            stop_reason = "no_candidate"
            break

        selected_g_lcb = max(
            float(best_abs_grad) - float(phase1_score_z_alpha_used) * float(best_grad_stderr),
            0.0,
        )
        for scout in candidate_gradient_scout:
            if int(scout.get("candidate_pool_index", -1)) == int(best_idx):
                scout["selected_for_optimization"] = True
        grad_conf = float(best_abs_grad / max(best_grad_std, 1e-12))
        grad_conf_stderr = float(best_abs_grad / max(best_grad_stderr, 1e-12))
        last_candidate_gradient_scout = [dict(row) for row in candidate_gradient_scout]
        last_sigma_hat = float(best_grad_stderr)
        last_g_lcb = float(selected_g_lcb)
        last_gradient_confidence = float(grad_conf)
        last_gradient_confidence_stderr = float(grad_conf_stderr)
        last_selected_pool_index = int(best_idx)
        last_selected_label = str(pool[int(best_idx)].label)
        if best_abs_grad < eps_grad:
            stop_reason = "eps_grad"
            break
        if grad_conf < min_conf:
            stop_reason = "low_gradient_confidence"
            break

        selected_ops.append(pool[best_idx])
        theta = np.append(theta, 0.0)
        if not allow_repeats:
            available.discard(best_idx)

        energy_prev = float(energy_current)
        objective_trace: list[dict[str, Any]] = []

        def _obj(x: np.ndarray) -> float:
            nonlocal nfev_total
            c = _adapt_ops_to_circuit(
                selected_ops,
                np.asarray(x, dtype=float),
                num_qubits=int(h_qop.num_qubits),
                reference_state=psi_ref,
            )
            est = noisy_oracle.evaluate(c, h_qop)
            nfev_total += 1
            objective_trace.append(
                {
                    "energy_noisy": float(est.mean),
                    "energy_noisy_mean": float(est.mean),
                    "energy_noisy_std": float(est.std),
                    "energy_noisy_stdev": float(est.stdev),
                    "energy_noisy_stderr": float(est.stderr),
                    "samples": int(est.n_samples),
                }
            )
            return float(est.mean)

        x0 = np.asarray(theta, dtype=float) + 0.02 * rng.normal(size=theta.size)
        if adapt_inner_key == "SPSA":
            spsa_res = spsa_minimize(
                fun=_obj,
                x0=x0,
                maxiter=int(args.adapt_maxiter),
                seed=int(args.adapt_seed) + int(depth),
                a=float(args.adapt_spsa_a),
                c=float(args.adapt_spsa_c),
                alpha=float(args.adapt_spsa_alpha),
                gamma=float(args.adapt_spsa_gamma),
                A=float(args.adapt_spsa_A),
                bounds=None,
                project="none",
                eval_repeats=int(args.adapt_spsa_eval_repeats),
                eval_agg=str(adapt_spsa_eval_agg),
                avg_last=int(args.adapt_spsa_avg_last),
            )
            theta = np.asarray(spsa_res.x, dtype=float)
            energy_current = float(spsa_res.fun)
            opt_nfev = int(spsa_res.nfev)
            opt_nit = int(spsa_res.nit)
            opt_success = bool(spsa_res.success)
            opt_message = str(spsa_res.message)
        else:
            assert minimize is not None
            res = minimize(
                _obj,
                x0,
                method="COBYLA",
                options={"maxiter": int(args.adapt_maxiter), "rhobeg": 0.3},
            )
            theta = np.asarray(res.x, dtype=float)
            energy_current = float(res.fun)
            opt_nfev = int(getattr(res, "nfev", 0))
            opt_nit = int(getattr(res, "nit", 0))
            opt_success = bool(getattr(res, "success", False))
            opt_message = str(getattr(res, "message", ""))
        delta_e = float(abs(energy_current - energy_prev))

        history_row = {
            "depth": int(depth + 1),
            "selected_pool_index": int(best_idx),
            "selected_label": str(pool[best_idx].label),
            "max_gradient": float(best_grad),
            "max_gradient_abs": float(best_abs_grad),
            "max_gradient_std": float(best_grad_std),
            "max_gradient_stderr": float(best_grad_stderr),
            "sigma_hat": float(best_grad_stderr),
            "g_lcb": float(selected_g_lcb),
            "gradient_confidence": float(grad_conf),
            "gradient_confidence_std": float(grad_conf),
            "gradient_confidence_stderr": float(grad_conf_stderr),
            "candidate_gradient_scout": [dict(row) for row in candidate_gradient_scout],
            "energy_before_opt": float(energy_prev),
            "energy_after_opt": float(energy_current),
            "delta_energy_abs": float(delta_e),
            "opt_method": str(adapt_inner_key),
            "opt_nfev": int(opt_nfev),
            "opt_nit": int(opt_nit),
            "opt_success": bool(opt_success),
            "opt_message": str(opt_message),
            "objective_trace": objective_trace,
        }
        if adapt_inner_key == "SPSA":
            history_row["spsa_params"] = dict(adapt_spsa_params)
        history.append(history_row)
        _ai_log(
            "hh_noise_adapt_iter_done",
            depth=int(depth + 1),
            selected_label=str(pool[best_idx].label),
            energy=float(energy_current),
            delta_e_abs=float(delta_e),
            gradient_abs=float(best_abs_grad),
            gradient_confidence=float(grad_conf),
            opt_method=str(adapt_inner_key),
        )

        if delta_e < eps_energy:
            stop_reason = "eps_energy"
            break
        if (not allow_repeats) and (not available):
            stop_reason = "pool_exhausted"
            break

    final_circuit = _adapt_ops_to_circuit(
        selected_ops,
        theta,
        num_qubits=int(h_qop.num_qubits),
        reference_state=psi_ref,
    )
    e_noisy = noisy_oracle.evaluate(final_circuit, h_qop)
    e_ideal = ideal_oracle.evaluate(final_circuit, h_qop)
    delta_noisy_minus_ideal = float(e_noisy.mean - e_ideal.mean)
    delta_noisy_minus_ideal_stderr = _combine_stderr(e_noisy.stderr, e_ideal.stderr)
    payload = {
        "success": True,
        "method": "adapt_vqe_noisy_oracle",
        "inner_optimizer": str(adapt_inner_key),
        "pool_type": str(args.adapt_pool),
        "pool_size": int(len(pool)),
        "allow_repeats": bool(allow_repeats),
        "ansatz_depth": int(len(selected_ops)),
        "num_parameters": int(theta.size),
        "operators": [str(op.label) for op in selected_ops],
        "optimal_point": [float(x) for x in theta.tolist()],
        "energy_noisy": float(e_noisy.mean),
        "energy_noisy_mean": float(e_noisy.mean),
        "energy_noisy_std": float(e_noisy.std),
        "energy_noisy_stdev": float(e_noisy.stdev),
        "energy_noisy_stderr": float(e_noisy.stderr),
        "energy_ideal_reference": float(e_ideal.mean),
        "energy_ideal_reference_mean": float(e_ideal.mean),
        "energy_ideal_reference_std": float(e_ideal.std),
        "energy_ideal_reference_stdev": float(e_ideal.stdev),
        "energy_ideal_reference_stderr": float(e_ideal.stderr),
        "delta_noisy_minus_ideal": float(delta_noisy_minus_ideal),
        "delta_noisy_minus_ideal_mean": float(delta_noisy_minus_ideal),
        "delta_noisy_minus_ideal_stderr": float(delta_noisy_minus_ideal_stderr),
        "stop_reason": str(stop_reason),
        "nfev_total": int(nfev_total),
        "history": history,
        "adapt_eps_grad": float(eps_grad),
        "adapt_eps_energy": float(eps_energy),
        "adapt_gradient_step": float(grad_step),
        "adapt_min_confidence": float(min_conf),
        "gradient_uncertainty_source": str(gradient_uncertainty_source),
        "phase1_score_z_alpha_used": float(phase1_score_z_alpha_used),
        "selection_metric_name": str(selection_metric_name),
        "gradient_confidence_mode_used": str(gradient_confidence_mode_used),
        "last_candidate_gradient_scout": [dict(row) for row in last_candidate_gradient_scout],
        "last_sigma_hat": float(last_sigma_hat),
        "last_g_lcb": float(last_g_lcb),
        "last_gradient_confidence": float(last_gradient_confidence),
        "last_gradient_confidence_stderr": float(last_gradient_confidence_stderr),
        "last_selected_pool_index": (
            int(last_selected_pool_index) if last_selected_pool_index is not None else None
        ),
        "last_selected_label": str(last_selected_label) if last_selected_label is not None else None,
        "elapsed_s": float(time.perf_counter() - t0),
    }
    if adapt_inner_key == "SPSA":
        payload["spsa"] = dict(adapt_spsa_params)
    return payload, selected_ops, theta


def _run_noisy_vqe(
    *,
    args: argparse.Namespace,
    ansatz: Any,
    psi_ref: np.ndarray,
    h_poly: Any,
    h_qop: SparsePauliOp,
    noisy_oracle: ExpectationOracle,
    ideal_oracle: ExpectationOracle,
) -> tuple[dict[str, Any], np.ndarray]:
    t0 = time.perf_counter()
    npar = int(getattr(ansatz, "num_parameters", 0))
    if npar <= 0:
        raise ValueError("ansatz has no free parameters")
    rng = np.random.default_rng(int(args.vqe_seed))

    history: list[dict[str, Any]] = []
    best_energy = float("inf")
    best_theta = np.zeros(npar, dtype=float)
    best_restart = -1
    best_nfev = 0
    best_nit = 0
    best_success = False
    best_message = "no run"
    method_key = str(args.vqe_method).strip().lower()
    noise_mode_key = str(args.noise_mode).strip().lower()
    backend_key = str(args.vqe_energy_backend).strip().lower()

    objective_source = "noisy_oracle"
    compiled_h = None
    if noise_mode_key == "ideal" and backend_key == "one_apply_compiled":
        compiled_h = compile_polynomial_action(
            h_poly,
            tol=1e-12,
        )
        objective_source = "compiled_one_apply_ideal"

    minimize = None
    if method_key != "spsa":
        try:
            from scipy.optimize import minimize as _scipy_minimize
        except Exception as exc:
            raise RuntimeError(
                "SciPy is required for this validation VQE loop when vqe-method is not SPSA. "
                "Install scipy or use --vqe-method SPSA."
            ) from exc
        minimize = _scipy_minimize

    vqe_spsa_eval_agg = str(args.vqe_spsa_eval_agg).strip().lower()
    vqe_spsa_params = {
        "a": float(args.vqe_spsa_a),
        "c": float(args.vqe_spsa_c),
        "alpha": float(args.vqe_spsa_alpha),
        "gamma": float(args.vqe_spsa_gamma),
        "A": float(args.vqe_spsa_A),
        "avg_last": int(args.vqe_spsa_avg_last),
        "eval_repeats": int(args.vqe_spsa_eval_repeats),
        "eval_agg": str(vqe_spsa_eval_agg),
    }

    for r in range(max(1, int(args.vqe_restarts))):
        x0 = 0.3 * rng.normal(size=npar)
        nfev_local = 0

        def _objective(x: np.ndarray) -> float:
            nonlocal nfev_local
            nfev_local += 1
            x_vec = np.asarray(x, dtype=float)
            if objective_source == "compiled_one_apply_ideal":
                assert compiled_h is not None
                psi = ansatz.prepare_state(x_vec, psi_ref)
                energy_obj, _hpsi = energy_via_one_apply(np.asarray(psi, dtype=complex), compiled_h)
                energy_mean = float(energy_obj)
                energy_std = 0.0
                energy_stdev = 0.0
                energy_stderr = 0.0
                oracle_samples = 0
            else:
                qc = _ansatz_to_circuit(
                    ansatz,
                    x_vec,
                    num_qubits=int(h_qop.num_qubits),
                    reference_state=psi_ref,
                )
                estimate = noisy_oracle.evaluate(qc, h_qop)
                energy_mean = float(estimate.mean)
                energy_std = float(estimate.std)
                energy_stdev = float(estimate.stdev)
                energy_stderr = float(estimate.stderr)
                oracle_samples = int(estimate.n_samples)
            history.append(
                {
                    "restart": int(r + 1),
                    "nfev_local": int(nfev_local),
                    "objective_source": str(objective_source),
                    "energy_objective": float(energy_mean),
                    "energy_noisy": float(energy_mean),
                    "energy_noisy_mean": float(energy_mean),
                    "energy_noisy_std": float(energy_std),
                    "energy_noisy_stdev": float(energy_stdev),
                    "energy_noisy_stderr": float(energy_stderr),
                    "oracle_samples": int(oracle_samples),
                }
            )
            return float(energy_mean)

        if method_key == "spsa":
            res = spsa_minimize(
                fun=_objective,
                x0=x0,
                maxiter=int(args.vqe_maxiter),
                seed=int(args.vqe_seed) + 1000 * int(r),
                a=float(args.vqe_spsa_a),
                c=float(args.vqe_spsa_c),
                alpha=float(args.vqe_spsa_alpha),
                gamma=float(args.vqe_spsa_gamma),
                A=float(args.vqe_spsa_A),
                bounds=None,
                project="none",
                eval_repeats=int(args.vqe_spsa_eval_repeats),
                eval_agg=str(vqe_spsa_eval_agg),
                avg_last=int(args.vqe_spsa_avg_last),
            )
            res_fun = float(res.fun)
            res_x = np.asarray(res.x, dtype=float)
            res_nfev = int(res.nfev)
            res_nit = int(res.nit)
            res_success = bool(res.success)
            res_message = str(res.message)
        else:
            assert minimize is not None
            res = minimize(
                _objective,
                x0,
                method=str(args.vqe_method),
                options={"maxiter": int(args.vqe_maxiter)},
            )
            res_fun = float(res.fun)
            res_x = np.asarray(res.x, dtype=float)
            res_nfev = int(getattr(res, "nfev", 0))
            res_nit = int(getattr(res, "nit", 0))
            res_success = bool(getattr(res, "success", False))
            res_message = str(getattr(res, "message", ""))

        if float(res_fun) < best_energy:
            best_energy = float(res_fun)
            best_theta = np.asarray(res_x, dtype=float)
            best_restart = int(r)
            best_nfev = int(res_nfev)
            best_nit = int(res_nit)
            best_success = bool(res_success)
            best_message = str(res_message)

        _ai_log(
            "hh_noise_vqe_restart_done",
            restart=int(r + 1),
            restarts=int(args.vqe_restarts),
            energy=float(res_fun),
            best_energy=float(best_energy),
        )

    best_circuit = _ansatz_to_circuit(
        ansatz,
        best_theta,
        num_qubits=int(h_qop.num_qubits),
        reference_state=psi_ref,
    )
    best_noisy = noisy_oracle.evaluate(best_circuit, h_qop)
    best_ideal = ideal_oracle.evaluate(best_circuit, h_qop)
    delta_noisy_minus_ideal = float(best_noisy.mean - best_ideal.mean)
    delta_noisy_minus_ideal_stderr = _combine_stderr(best_noisy.stderr, best_ideal.stderr)

    payload = {
        "success": bool(best_success),
        "method": "noisy_vqe_qiskit_oracle",
        "ansatz": str(args.ansatz),
        "optimizer_method": str(args.vqe_method),
        "objective_source": str(objective_source),
        "energy_backend": str(args.vqe_energy_backend),
        "reps": int(args.vqe_reps),
        "restarts": int(args.vqe_restarts),
        "maxiter": int(args.vqe_maxiter),
        "num_parameters": int(npar),
        "best_restart": int(best_restart + 1),
        "nfev": int(best_nfev),
        "nit": int(best_nit),
        "message": str(best_message),
        "energy_noisy": float(best_noisy.mean),
        "energy_noisy_mean": float(best_noisy.mean),
        "energy_noisy_std": float(best_noisy.std),
        "energy_noisy_stdev": float(best_noisy.stdev),
        "energy_noisy_stderr": float(best_noisy.stderr),
        "energy_ideal_reference": float(best_ideal.mean),
        "energy_ideal_reference_mean": float(best_ideal.mean),
        "energy_ideal_reference_std": float(best_ideal.std),
        "energy_ideal_reference_stdev": float(best_ideal.stdev),
        "energy_ideal_reference_stderr": float(best_ideal.stderr),
        "delta_noisy_minus_ideal": float(delta_noisy_minus_ideal),
        "delta_noisy_minus_ideal_mean": float(delta_noisy_minus_ideal),
        "delta_noisy_minus_ideal_stderr": float(delta_noisy_minus_ideal_stderr),
        "optimal_point": [float(x) for x in best_theta.tolist()],
        "objective_history": history,
        "elapsed_s": float(time.perf_counter() - t0),
    }
    if method_key == "spsa":
        payload["spsa"] = dict(vqe_spsa_params)
    return payload, best_theta


def _run_noisy_trotter(
    *,
    args: argparse.Namespace,
    initial_circuit: QuantumCircuit,
    ordered_qop: SparsePauliOp,
    observables: dict[str, SparsePauliOp],
    noisy_oracle: ExpectationOracle,
    ideal_oracle: ExpectationOracle,
) -> list[dict[str, Any]]:
    times = np.linspace(0.0, float(args.t_final), int(args.num_times))
    rows: list[dict[str, Any]] = []
    total = max(1, int(times.size))
    stride = max(1, total // 10)

    for idx, t_val in enumerate(times):
        qc_t = _trotterized_circuit(
            initial_circuit,
            ordered_qop,
            time_value=float(t_val),
            trotter_steps=int(args.trotter_steps),
            suzuki_order=int(args.suzuki_order),
        )
        row: dict[str, Any] = {"time": float(t_val)}
        for key, obs in observables.items():
            noisy = noisy_oracle.evaluate(qc_t, obs)
            ideal = ideal_oracle.evaluate(qc_t, obs)
            delta_mean = float(noisy.mean - ideal.mean)
            delta_stderr = _combine_stderr(noisy.stderr, ideal.stderr)
            row[f"{key}_noisy"] = float(noisy.mean)
            row[f"{key}_noisy_mean"] = float(noisy.mean)
            row[f"{key}_noisy_std"] = float(noisy.std)
            row[f"{key}_noisy_stdev"] = float(noisy.stdev)
            row[f"{key}_noisy_stderr"] = float(noisy.stderr)
            row[f"{key}_ideal"] = float(ideal.mean)
            row[f"{key}_ideal_mean"] = float(ideal.mean)
            row[f"{key}_ideal_std"] = float(ideal.std)
            row[f"{key}_ideal_stdev"] = float(ideal.stdev)
            row[f"{key}_ideal_stderr"] = float(ideal.stderr)
            row[f"{key}_delta_noisy_minus_ideal"] = float(delta_mean)
            row[f"{key}_delta_noisy_minus_ideal_mean"] = float(delta_mean)
            row[f"{key}_delta_noisy_minus_ideal_stderr"] = float(delta_stderr)
        rows.append(row)

        if idx % stride == 0 or idx == total - 1:
            _ai_log(
                "hh_noise_trotter_progress",
                step=int(idx + 1),
                total_steps=int(total),
                t=float(t_val),
            )
    return rows


def _trajectory_delta_uncertainty(trajectory_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    if not trajectory_rows:
        return {}
    keys = [str(k) for k in trajectory_rows[0].keys() if str(k).endswith("_delta_noisy_minus_ideal")]
    out: dict[str, dict[str, float]] = {}
    for delta_key in keys:
        stderr_key = f"{delta_key}_stderr"
        delta = np.asarray([float(r.get(delta_key, float("nan"))) for r in trajectory_rows], dtype=float)
        stderr = np.asarray([float(r.get(stderr_key, float("nan"))) for r in trajectory_rows], dtype=float)
        observable = str(delta_key).replace("_delta_noisy_minus_ideal", "")
        out[observable] = _delta_uncertainty_metrics(delta, stderr)
    return out


def _initial_state_selection(
    *,
    args: argparse.Namespace,
    ansatz: Any,
    psi_ref: np.ndarray,
    hmat: np.ndarray,
    best_theta: np.ndarray | None,
    adapt_ops: list[AnsatzTerm] | None,
    adapt_theta: np.ndarray | None,
) -> tuple[str, QuantumCircuit]:
    source = str(args.initial_state_source).strip().lower()
    num_qubits = int(round(math.log2(int(hmat.shape[0]))))
    if source == "vqe":
        if best_theta is None:
            raise ValueError(
                "initial_state_source=vqe requires a VQE run in this invocation."
            )
        qc = _ansatz_to_circuit(
            ansatz,
            best_theta,
            num_qubits=num_qubits,
            reference_state=psi_ref,
        )
        return "vqe", qc
    if source == "adapt":
        if (adapt_ops is None) or (adapt_theta is None):
            raise ValueError(
                "initial_state_source=adapt requires --run-adapt in this invocation."
            )
        qc = _adapt_ops_to_circuit(
            adapt_ops,
            np.asarray(adapt_theta, dtype=float),
            num_qubits=num_qubits,
            reference_state=psi_ref,
        )
        return "adapt", qc
    if source == "hf":
        qc = QuantumCircuit(num_qubits)
        from pipelines.exact_bench.noise_oracle_runtime import _append_reference_state  # local import

        _append_reference_state(qc, psi_ref)
        return "hf", qc

    evals, evecs = np.linalg.eigh(np.asarray(hmat, dtype=complex))
    psi_exact = np.asarray(evecs[:, int(np.argmin(np.real(evals)))], dtype=complex).reshape(-1)
    qc = QuantumCircuit(num_qubits)
    from pipelines.exact_bench.noise_oracle_runtime import _append_reference_state  # local import

    _append_reference_state(qc, psi_exact)
    return "exact", qc


def _parse_compare_observables(raw: str) -> list[str]:
    valid = {
        "energy_static_trotter",
        "doublon_trotter",
        "n_up_site0_trotter",
        "n_dn_site0_trotter",
    }
    parts = [str(x).strip() for x in str(raw).split(",")]
    obs = [x for x in parts if x]
    if not obs:
        raise ValueError(
            "compare_observables must contain at least one observable from: "
            "energy_static_trotter,doublon_trotter,n_up_site0_trotter,n_dn_site0_trotter"
        )
    bad = [x for x in obs if x not in valid]
    if bad:
        raise ValueError(
            "Unsupported compare_observables entries: "
            f"{bad}. Allowed: {sorted(valid)}"
        )
    return obs


def _load_legacy_trajectory(
    legacy_json: Path,
    observables: list[str],
) -> dict[str, Any]:
    payload = json.loads(Path(legacy_json).read_text(encoding="utf-8"))
    traj = payload.get("trajectory")
    if not isinstance(traj, list) or not traj:
        raise ValueError(
            f"Legacy reference JSON '{legacy_json}' must contain a non-empty trajectory list."
        )
    times: list[float] = []
    series: dict[str, list[float]] = {obs: [] for obs in observables}
    for i, row in enumerate(traj):
        if not isinstance(row, dict):
            raise ValueError(f"Legacy trajectory row {i} is not a dict.")
        if "time" not in row:
            raise ValueError(f"Legacy trajectory row {i} missing 'time'.")
        times.append(float(row["time"]))
        for obs in observables:
            if obs not in row:
                raise ValueError(
                    f"Legacy trajectory row {i} missing observable '{obs}'."
                )
            series[obs].append(float(row[obs]))
    return {
        "reference_json": str(legacy_json),
        "times": np.asarray(times, dtype=float),
        "series": {k: np.asarray(v, dtype=float) for k, v in series.items()},
    }


def _compute_legacy_parity(
    *,
    legacy_ref: dict[str, Any],
    new_trajectory: list[dict[str, Any]],
    observables: list[str],
    tolerance: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "skipped": False,
        "reference_json": str(legacy_ref.get("reference_json", "")),
        "observables": list(observables),
        "tolerance": float(tolerance),
        "time_grid_match": False,
        "reason": "",
        "per_observable": {},
        "passed_all": False,
    }
    if not new_trajectory:
        result["reason"] = "No new trajectory rows were produced."
        return result

    new_times = np.asarray([float(r["time"]) for r in new_trajectory], dtype=float)
    legacy_times = np.asarray(legacy_ref["times"], dtype=float)
    time_grid_match = bool(np.array_equal(new_times, legacy_times))
    result["time_grid_match"] = time_grid_match
    if not time_grid_match:
        result["reason"] = (
            f"time-grid mismatch (new={len(new_times)} points, legacy={len(legacy_times)} points)"
        )

    per_obs: dict[str, Any] = {}
    passed_all = bool(time_grid_match)
    for obs in observables:
        new_key = f"{obs}_ideal"
        new_vals = np.asarray([float(r[new_key]) for r in new_trajectory], dtype=float)
        legacy_vals = np.asarray(legacy_ref["series"][obs], dtype=float)
        if len(new_vals) != len(legacy_vals):
            max_abs = float("inf")
            mean_abs = float("inf")
            final_abs = float("inf")
            passed = False
        else:
            deltas = np.abs(new_vals - legacy_vals)
            max_abs = float(np.max(deltas))
            mean_abs = float(np.mean(deltas))
            final_abs = float(deltas[-1])
            passed = bool(time_grid_match and (max_abs <= float(tolerance)))
        per_obs[obs] = {
            "max_abs_delta": max_abs,
            "mean_abs_delta": mean_abs,
            "final_abs_delta": final_abs,
            "passed": bool(passed),
        }
        passed_all = bool(passed_all and passed)
    result["per_observable"] = per_obs
    result["passed_all"] = bool(passed_all)
    return result


def _write_legacy_comparison_plot(
    *,
    plot_path: Path,
    payload: dict[str, Any],
    legacy_ref: dict[str, Any] | None,
) -> None:
    require_matplotlib()
    plt = get_plt()

    trajectory = payload.get("trajectory", [])
    if not trajectory:
        raise ValueError("Cannot write comparison plot: payload has no trajectory.")
    times = np.asarray([float(r["time"]) for r in trajectory], dtype=float)
    e_noisy = np.asarray([float(r["energy_static_trotter_noisy"]) for r in trajectory], dtype=float)
    e_ideal = np.asarray([float(r["energy_static_trotter_ideal"]) for r in trajectory], dtype=float)
    d_noisy = np.asarray([float(r["doublon_trotter_noisy"]) for r in trajectory], dtype=float)
    d_ideal = np.asarray([float(r["doublon_trotter_ideal"]) for r in trajectory], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.0), sharex=False)
    axes[0].plot(times, e_noisy, label="new noisy", color="#1f77b4")
    axes[0].plot(times, e_ideal, label="new ideal", color="#ff7f0e", linestyle="--")
    if legacy_ref is not None:
        lt = np.asarray(legacy_ref["times"], dtype=float)
        le = np.asarray(legacy_ref["series"]["energy_static_trotter"], dtype=float)
        axes[0].plot(lt, le, label="legacy", color="#2ca02c", linestyle=":")
    axes[0].set_ylabel("Energy")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="best")

    axes[1].plot(times, d_noisy, label="new noisy", color="#9467bd")
    axes[1].plot(times, d_ideal, label="new ideal", color="#d62728", linestyle="--")
    if legacy_ref is not None:
        lt = np.asarray(legacy_ref["times"], dtype=float)
        ld = np.asarray(legacy_ref["series"]["doublon_trotter"], dtype=float)
        axes[1].plot(lt, ld, label="legacy", color="#8c564b", linestyle=":")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Doublon")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle("HH L2 Comparison: Noisy vs Noiseless vs Legacy")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(plot_path), dpi=170)
    plt.close(fig)


def _write_noise_validation_pdf(
    *,
    pdf_path: Path,
    payload: dict[str, Any],
) -> None:
    require_matplotlib()
    plt = get_plt()
    PdfPages = get_PdfPages()

    settings = payload.get("settings", {})
    vqe = payload.get("vqe", {})
    trajectory = payload.get("trajectory", [])
    backend = payload.get("backend", {})
    noise = payload.get("noise_config", {})
    fallback = payload.get("execution_fallback", {})
    adapt = payload.get("adapt", {})
    legacy_parity = payload.get("legacy_parity", {})
    final = trajectory[-1] if trajectory else {}
    noise_caption = (
        "noise_config: "
        f"mode={noise.get('noise_mode')}, "
        f"shots={noise.get('shots')}, "
        f"oracle_repeats={noise.get('oracle_repeats')}, "
        f"oracle_aggregate={noise.get('oracle_aggregate')}, "
        f"{_mitigation_caption(noise.get('mitigation', {}), noise.get('symmetry_mitigation', {}))}"
    )
    manifest_sections: list[tuple[str, list[tuple[str, Any]]]] = [
        (
            "Model and regime",
            [
                ("Model family", str(payload.get("model", "Hubbard-Holstein"))),
                ("Problem", settings.get("problem")),
                ("Ansatz", settings.get("ansatz")),
                ("L", settings.get("L")),
                ("Boundary", settings.get("boundary")),
                ("Ordering", settings.get("ordering")),
            ],
        ),
        (
            "Noise and backend",
            [
                ("Noise mode", noise.get("noise_mode")),
                ("shots", noise.get("shots")),
                ("oracle_repeats", noise.get("oracle_repeats")),
                ("oracle_aggregate", noise.get("oracle_aggregate")),
                ("Mitigation", noise.get("mitigation")),
                ("Backend", backend.get("backend_name")),
                ("Using fake backend", backend.get("using_fake_backend")),
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
            "Validation workload",
            [
                ("Run VQE", settings.get("run_vqe")),
                ("Run Trotter", settings.get("run_trotter")),
                ("Run ADAPT", settings.get("run_adapt")),
                ("Initial state source", settings.get("initial_state_source")),
            ],
        ),
    ]

    summary_sections: list[tuple[str, list[tuple[str, Any]]]] = [
        (
            "Validation verdict",
            [
                ("Fallback used", fallback.get("used")),
                ("Fallback mode", fallback.get("mode")),
                ("Legacy parity enabled", not bool(legacy_parity.get("skipped", True))),
                ("Legacy parity passed", legacy_parity.get("passed_all")),
                ("Legacy parity time-grid match", legacy_parity.get("time_grid_match")),
            ],
        ),
        (
            "VQE effect summary",
            [
                ("Noisy energy", vqe.get("energy_noisy")),
                ("Noisy energy stderr", vqe.get("energy_noisy_stderr")),
                ("Ideal reference energy", vqe.get("energy_ideal_reference")),
                ("Noisy-ideal delta", vqe.get("delta_noisy_minus_ideal")),
                ("Noisy-ideal delta stderr", vqe.get("delta_noisy_minus_ideal_stderr")),
            ],
        ),
        (
            "Final trajectory point",
            [
                ("Energy delta", final.get("energy_static_trotter_delta_noisy_minus_ideal")),
                ("Energy delta stderr", final.get("energy_static_trotter_delta_noisy_minus_ideal_stderr")),
                ("Doublon delta", final.get("doublon_trotter_delta_noisy_minus_ideal")),
                ("Doublon delta stderr", final.get("doublon_trotter_delta_noisy_minus_ideal_stderr")),
            ],
        ),
        (
            "ADAPT summary",
            [
                ("Enabled", settings.get("run_adapt")),
                ("Success", adapt.get("success")),
                ("Depth", adapt.get("ansatz_depth")),
                ("Stop reason", adapt.get("stop_reason")),
            ],
        ),
    ]

    detail_lines = [
        "HH/Hubbard Noise Validation Summary",
        "",
        f"problem: {settings.get('problem')}",
        f"ansatz: {settings.get('ansatz')}",
        f"noise_mode: {noise.get('noise_mode')}",
        noise_caption,
        f"backend: {backend.get('backend_name')}",
        f"fallback_used: {fallback.get('used')}",
        f"fallback_mode: {fallback.get('mode')}",
        f"fallback_reason: {str(fallback.get('reason', ''))[:140]}",
        "",
        "Legacy parity (vs pre-noise baseline):",
        f"  enabled: {not bool(legacy_parity.get('skipped', True))}",
        f"  passed_all: {legacy_parity.get('passed_all')}",
        f"  time_grid_match: {legacy_parity.get('time_grid_match')}",
        f"  tolerance: {legacy_parity.get('tolerance')}",
        f"  reference_json: {legacy_parity.get('reference_json')}",
        "",
        "VQE:",
        f"  success: {vqe.get('success')}",
        f"  optimizer: {vqe.get('optimizer_method')}",
        f"  objective_source: {vqe.get('objective_source')}",
        f"  energy_backend: {vqe.get('energy_backend')}",
        f"  noisy energy: {vqe.get('energy_noisy')}",
        f"  noisy energy stderr: {vqe.get('energy_noisy_stderr')}",
        f"  ideal reference energy: {vqe.get('energy_ideal_reference')}",
        f"  noisy-ideal delta: {vqe.get('delta_noisy_minus_ideal')}",
        f"  noisy-ideal delta stderr: {vqe.get('delta_noisy_minus_ideal_stderr')}",
        "",
        "ADAPT (phase 2):",
        f"  enabled: {settings.get('run_adapt')}",
        f"  success: {adapt.get('success')}",
        f"  inner optimizer: {adapt.get('inner_optimizer')}",
        f"  depth: {adapt.get('ansatz_depth')}",
        f"  stop_reason: {adapt.get('stop_reason')}",
        f"  noisy-ideal delta: {adapt.get('delta_noisy_minus_ideal')}",
        f"  noisy-ideal delta stderr: {adapt.get('delta_noisy_minus_ideal_stderr')}",
        "",
        "Final trajectory point:",
        f"  energy noisy: {final.get('energy_static_trotter_noisy')}",
        f"  energy ideal: {final.get('energy_static_trotter_ideal')}",
        f"  energy delta: {final.get('energy_static_trotter_delta_noisy_minus_ideal')}",
        f"  energy delta stderr: {final.get('energy_static_trotter_delta_noisy_minus_ideal_stderr')}",
        f"  doublon noisy: {final.get('doublon_trotter_noisy')}",
        f"  doublon ideal: {final.get('doublon_trotter_ideal')}",
        f"  doublon delta: {final.get('doublon_trotter_delta_noisy_minus_ideal')}",
        f"  doublon delta stderr: {final.get('doublon_trotter_delta_noisy_minus_ideal_stderr')}",
    ]
    with PdfPages(str(pdf_path)) as pdf:
        render_manifest_overview_page(
            pdf,
            title=f"{payload.get('model', 'HH/Hubbard')} noise validation — L={settings.get('L')}",
            experiment_statement=(
                "Validation run comparing noisy and ideal observables, with optional parity checks against the legacy baseline."
            ),
            sections=manifest_sections,
            notes=[
                "Noise/backend/fallback details are summarized up front; full audit detail appears later.",
                "The full executed command appears in the appendix.",
            ],
        )
        render_executive_summary_page(
            pdf,
            title="Executive summary",
            experiment_statement="Headline verdict first: validation status, noisy-minus-ideal effect size, and final observable deltas.",
            sections=summary_sections,
            notes=[
                noise_caption,
            ],
        )
        render_section_divider_page(
            pdf,
            title="Scientific effect pages",
            summary="These pages foreground how noise changed the observables before backend and parity audit detail.",
            bullets=[
                "Noisy vs ideal trajectories.",
                "Noisy-minus-ideal delta bands.",
                "Compact VQE scoreboard.",
            ],
        )
        vqe_spsa = vqe.get("spsa")
        if isinstance(vqe_spsa, dict):
            detail_lines.extend(
                [
                    "",
                    "VQE SPSA params:",
                    f"  a={vqe_spsa.get('a')} c={vqe_spsa.get('c')} A={vqe_spsa.get('A')}",
                    f"  alpha={vqe_spsa.get('alpha')} gamma={vqe_spsa.get('gamma')}",
                    (
                        "  eval_repeats={eval_repeats} eval_agg={eval_agg} avg_last={avg_last}".format(
                            eval_repeats=vqe_spsa.get("eval_repeats"),
                            eval_agg=vqe_spsa.get("eval_agg"),
                            avg_last=vqe_spsa.get("avg_last"),
                        )
                    ),
                ]
            )
        adapt_spsa = adapt.get("spsa")
        if isinstance(adapt_spsa, dict):
            detail_lines.extend(
                [
                    "",
                    "ADAPT SPSA params:",
                    f"  a={adapt_spsa.get('a')} c={adapt_spsa.get('c')} A={adapt_spsa.get('A')}",
                    f"  alpha={adapt_spsa.get('alpha')} gamma={adapt_spsa.get('gamma')}",
                    (
                        "  eval_repeats={eval_repeats} eval_agg={eval_agg} avg_last={avg_last}".format(
                            eval_repeats=adapt_spsa.get("eval_repeats"),
                            eval_agg=adapt_spsa.get("eval_agg"),
                            avg_last=adapt_spsa.get("avg_last"),
                        )
                    ),
                ]
            )
        per_obs = legacy_parity.get("per_observable", {})
        obs_order = legacy_parity.get("observables", [])
        if isinstance(per_obs, dict) and isinstance(obs_order, list):
            detail_lines.append("")
            detail_lines.append("Legacy parity deltas:")
            for obs in obs_order:
                if obs not in per_obs:
                    continue
                rec = per_obs.get(obs, {})
                detail_lines.append(
                    f"  {obs}: max={rec.get('max_abs_delta')} "
                    f"mean={rec.get('mean_abs_delta')} final={rec.get('final_abs_delta')} "
                    f"passed={rec.get('passed')}"
                )

        if trajectory:
            ts = np.array([float(r["time"]) for r in trajectory], dtype=float)
            e_noisy = np.array([float(r["energy_static_trotter_noisy"]) for r in trajectory], dtype=float)
            e_ideal = np.array([float(r["energy_static_trotter_ideal"]) for r in trajectory], dtype=float)
            e_delta = np.array(
                [float(r["energy_static_trotter_delta_noisy_minus_ideal"]) for r in trajectory], dtype=float
            )
            e_delta_stderr = np.array(
                [float(r.get("energy_static_trotter_delta_noisy_minus_ideal_stderr", 0.0)) for r in trajectory],
                dtype=float,
            )
            d_noisy = np.array([float(r["doublon_trotter_noisy"]) for r in trajectory], dtype=float)
            d_ideal = np.array([float(r["doublon_trotter_ideal"]) for r in trajectory], dtype=float)
            d_delta = np.array(
                [float(r["doublon_trotter_delta_noisy_minus_ideal"]) for r in trajectory], dtype=float
            )
            d_delta_stderr = np.array(
                [float(r.get("doublon_trotter_delta_noisy_minus_ideal_stderr", 0.0)) for r in trajectory],
                dtype=float,
            )

            fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)
            axes[0].plot(ts, e_noisy, label="energy noisy", color="#1f77b4")
            axes[0].plot(ts, e_ideal, label="energy ideal", color="#ff7f0e", linestyle="--")
            axes[0].set_ylabel("Energy")
            axes[0].grid(alpha=0.25)
            axes[0].legend(fontsize=8, loc="best")

            axes[1].plot(ts, d_noisy, label="doublon noisy", color="#2ca02c")
            axes[1].plot(ts, d_ideal, label="doublon ideal", color="#d62728", linestyle="--")
            axes[1].set_xlabel("Time")
            axes[1].set_ylabel("Doublon")
            axes[1].grid(alpha=0.25)
            axes[1].legend(fontsize=8, loc="best")
            fig.suptitle("Noisy vs Ideal Trajectory")
            fig.text(0.01, 0.01, noise_caption, fontsize=8, ha="left", va="bottom")
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
            pdf.savefig(fig)
            plt.close(fig)

            fig_delta, axes_delta = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)
            axes_delta[0].plot(ts, e_delta, color="#d62728", linewidth=1.2, label="ΔE (noisy-ideal)")
            axes_delta[0].fill_between(
                ts,
                e_delta - (2.0 * e_delta_stderr),
                e_delta + (2.0 * e_delta_stderr),
                color="#d62728",
                alpha=0.18,
                linewidth=0.0,
                label="±2 stderr",
            )
            axes_delta[0].axhline(0.0, color="#111111", linewidth=0.8, linestyle=":")
            axes_delta[0].set_ylabel("ΔEnergy")
            axes_delta[0].grid(alpha=0.25)
            axes_delta[0].legend(fontsize=8, loc="best")

            axes_delta[1].plot(ts, d_delta, color="#9467bd", linewidth=1.2, label="ΔDoublon (noisy-ideal)")
            axes_delta[1].fill_between(
                ts,
                d_delta - (2.0 * d_delta_stderr),
                d_delta + (2.0 * d_delta_stderr),
                color="#9467bd",
                alpha=0.18,
                linewidth=0.0,
                label="±2 stderr",
            )
            axes_delta[1].axhline(0.0, color="#111111", linewidth=0.8, linestyle=":")
            axes_delta[1].set_xlabel("Time")
            axes_delta[1].set_ylabel("ΔDoublon")
            axes_delta[1].grid(alpha=0.25)
            axes_delta[1].legend(fontsize=8, loc="best")
            fig_delta.suptitle("Noisy-Ideal Delta with Uncertainty Bands")
            fig_delta.text(0.01, 0.01, noise_caption, fontsize=8, ha="left", va="bottom")
            fig_delta.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
            pdf.savefig(fig_delta)
            plt.close(fig_delta)

        if vqe:
            rows = [
                ["Noisy VQE", f"{float(vqe.get('energy_noisy', float('nan'))):.8f}"],
                ["Noisy VQE stderr", f"{float(vqe.get('energy_noisy_stderr', float('nan'))):.3e}"],
                ["Ideal reference", f"{float(vqe.get('energy_ideal_reference', float('nan'))):.8f}"],
                ["Noisy-Ideal", f"{float(vqe.get('delta_noisy_minus_ideal', float('nan'))):.3e}"],
                ["Noisy-Ideal stderr", f"{float(vqe.get('delta_noisy_minus_ideal_stderr', float('nan'))):.3e}"],
                ["#params", str(vqe.get("num_parameters"))],
                ["best restart", str(vqe.get("best_restart"))],
            ]
            fig_tbl, ax_tbl = plt.subplots(figsize=(8.5, 4.5))
            render_compact_table(
                ax_tbl,
                title="VQE Scoreboard (Noisy Oracle)",
                col_labels=["Metric", "Value"],
                rows=rows,
            )
            fig_tbl.tight_layout()
            pdf.savefig(fig_tbl)
            plt.close(fig_tbl)

        render_section_divider_page(
            pdf,
            title="Technical appendix",
            summary="Backend fallback detail, mitigation configuration, legacy parity audit, and reproducibility information.",
            bullets=[
                "Detailed validation summary.",
                "Legacy parity breakdown by observable.",
                "Executed command.",
            ],
        )
        render_text_page(pdf, detail_lines, fontsize=9)
        render_command_page(
            pdf,
            str(payload.get("run_command", "")),
            script_name="pipelines/exact_bench/hh_noise_hardware_validation.py",
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HH-first noisy/hardware feasibility validation runner.")
    p.add_argument("--problem", choices=["hh", "hubbard"], default="hh")
    p.add_argument("--ansatz", choices=["hh_hva", "hh_hva_tw", "hh_hva_ptw", "hva", "uccsd"], default="hh_hva")
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--u", type=float, default=4.0)
    p.add_argument("--dv", type=float, default=0.0)
    p.add_argument("--omega0", type=float, default=1.0)
    p.add_argument("--g-ep", type=float, default=0.5)
    p.add_argument("--n-ph-max", type=int, default=1)
    p.add_argument("--boson-encoding", choices=["binary"], default="binary")
    p.add_argument("--boundary", choices=["periodic", "open"], default="periodic")
    p.add_argument("--ordering", choices=["blocked", "interleaved"], default="blocked")

    p.add_argument("--noise-mode", choices=["ideal", "shots", "aer_noise", "runtime"], default="ideal")
    p.add_argument("--shots", type=int, default=2048)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--oracle-repeats", type=int, default=1)
    p.add_argument("--oracle-aggregate", choices=["mean", "median"], default="mean")
    p.add_argument("--mitigation", choices=["none", "readout", "zne", "dd"], default="none")
    p.add_argument(
        "--symmetry-mitigation-mode",
        choices=["off", "verify_only", "postselect_diag_v1", "projector_renorm_v1"],
        default="off",
    )
    p.add_argument("--zne-scales", type=str, default=None)
    p.add_argument("--dd-sequence", type=str, default=None)
    p.add_argument("--backend-name", type=str, default=None)
    p.add_argument("--use-fake-backend", action="store_true")
    p.set_defaults(allow_aer_fallback=True)
    p.add_argument("--allow-aer-fallback", dest="allow_aer_fallback", action="store_true")
    p.add_argument("--no-allow-aer-fallback", dest="allow_aer_fallback", action="store_false")
    p.set_defaults(omp_shm_workaround=True)
    p.add_argument("--omp-shm-workaround", dest="omp_shm_workaround", action="store_true")
    p.add_argument("--no-omp-shm-workaround", dest="omp_shm_workaround", action="store_false")

    p.set_defaults(run_vqe=True, run_trotter=True)
    p.add_argument("--run-vqe", dest="run_vqe", action="store_true")
    p.add_argument("--no-run-vqe", dest="run_vqe", action="store_false")
    p.add_argument("--run-trotter", dest="run_trotter", action="store_true")
    p.add_argument("--no-run-trotter", dest="run_trotter", action="store_false")
    p.set_defaults(run_adapt=False)
    p.add_argument("--run-adapt", dest="run_adapt", action="store_true")
    p.add_argument("--no-run-adapt", dest="run_adapt", action="store_false")
    p.add_argument("--initial-state-source", choices=["hf", "vqe", "adapt", "exact"], default="vqe")

    p.add_argument("--vqe-reps", type=int, default=None)
    p.add_argument("--vqe-restarts", type=int, default=None)
    p.add_argument("--vqe-maxiter", type=int, default=None)
    p.add_argument(
        "--vqe-method",
        type=str,
        choices=["SLSQP", "COBYLA", "L-BFGS-B", "Powell", "Nelder-Mead", "SPSA"],
        default=None,
    )
    p.add_argument("--vqe-seed", type=int, default=7)
    p.add_argument(
        "--vqe-energy-backend",
        type=str,
        default="legacy",
        choices=["legacy", "one_apply_compiled"],
    )
    p.add_argument("--vqe-spsa-a", type=float, default=0.2)
    p.add_argument("--vqe-spsa-c", type=float, default=0.1)
    p.add_argument("--vqe-spsa-alpha", type=float, default=0.602)
    p.add_argument("--vqe-spsa-gamma", type=float, default=0.101)
    p.add_argument("--vqe-spsa-A", type=float, default=10.0)
    p.add_argument("--vqe-spsa-avg-last", type=int, default=0)
    p.add_argument("--vqe-spsa-eval-repeats", type=int, default=1)
    p.add_argument("--vqe-spsa-eval-agg", choices=["mean", "median"], default="mean")

    p.add_argument("--adapt-pool", choices=["hva", "uccsd", "full_hamiltonian"], default="hva")
    p.add_argument("--adapt-max-depth", type=int, default=20)
    p.add_argument("--adapt-eps-grad", type=float, default=1e-5)
    p.add_argument("--adapt-eps-energy", type=float, default=1e-8)
    p.add_argument("--adapt-maxiter", type=int, default=300)
    p.add_argument("--adapt-seed", type=int, default=7)
    p.add_argument("--adapt-inner-optimizer", choices=["COBYLA", "SPSA"], default="SPSA")
    p.add_argument("--adapt-spsa-a", type=float, default=0.2)
    p.add_argument("--adapt-spsa-c", type=float, default=0.1)
    p.add_argument("--adapt-spsa-alpha", type=float, default=0.602)
    p.add_argument("--adapt-spsa-gamma", type=float, default=0.101)
    p.add_argument("--adapt-spsa-A", type=float, default=10.0)
    p.add_argument("--adapt-spsa-avg-last", type=int, default=0)
    p.add_argument("--adapt-spsa-eval-repeats", type=int, default=1)
    p.add_argument("--adapt-spsa-eval-agg", choices=["mean", "median"], default="mean")
    p.set_defaults(adapt_allow_repeats=True)
    p.add_argument("--adapt-allow-repeats", dest="adapt_allow_repeats", action="store_true")
    p.add_argument("--adapt-no-repeats", dest="adapt_allow_repeats", action="store_false")
    p.add_argument("--adapt-gradient-step", type=float, default=0.1)
    p.add_argument("--adapt-min-confidence", type=float, default=0.0)

    p.add_argument("--t-final", type=float, default=None)
    p.add_argument("--num-times", type=int, default=None)
    p.add_argument("--suzuki-order", type=int, default=2)
    p.add_argument("--trotter-steps", type=int, default=None)
    p.add_argument("--exact-steps-multiplier", type=int, default=None)

    p.add_argument(
        "--smoke-test-intentionally-weak",
        action="store_true",
        help="# SMOKE TEST - intentionally weak settings",
    )
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-pdf", type=Path, default=None)
    p.add_argument("--legacy-reference-json", type=Path, default=None)
    p.add_argument("--legacy-parity-tol", type=float, default=1e-10)
    p.add_argument("--output-compare-plot", type=Path, default=None)
    p.add_argument(
        "--compare-observables",
        type=str,
        default="energy_static_trotter,doublon_trotter",
    )
    p.add_argument("--skip-pdf", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _apply_defaults_and_minimums(parse_args(argv))
    compare_observables = _parse_compare_observables(str(args.compare_observables))
    if str(args.problem).strip().lower() == "hubbard" and str(args.adapt_pool).strip().lower() == "hva":
        _ai_log(
            "hh_noise_adapt_pool_default_override",
            problem="hubbard",
            from_pool="hva",
            to_pool="uccsd",
        )
        args.adapt_pool = "uccsd"
    _ai_log("hh_noise_validation_start", settings=vars(args))

    artifacts_dir = REPO_ROOT / "artifacts"
    json_dir = artifacts_dir / "json"
    pdf_dir = artifacts_dir / "pdf"
    json_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    tag = f"L{int(args.L)}_{str(args.problem)}_{str(args.ansatz)}_{str(args.noise_mode)}"
    output_json = args.output_json or (json_dir / f"hh_noise_validation_{tag}.json")
    output_pdf = args.output_pdf or (pdf_dir / f"hh_noise_validation_{tag}.pdf")

    num_particles = _half_filled_particles(int(args.L))
    h_poly = _build_hamiltonian(args)
    h_qop = _pauli_poly_to_sparse_pauli_op(h_poly)
    hmat = hamiltonian_matrix(h_poly)
    native_order, coeff_map_exyz = _collect_hardcoded_terms_exyz(h_poly)
    ordered_labels_exyz = sorted(native_order)
    ordered_qop = _ordered_qop_from_exyz(ordered_labels_exyz, coeff_map_exyz)

    if str(args.problem).strip().lower() == "hh":
        exact_filtered = float(
            exact_ground_energy_sector_hh(
                h_poly,
                num_sites=int(args.L),
                num_particles=num_particles,
                n_ph_max=int(args.n_ph_max),
                boson_encoding=str(args.boson_encoding),
                indexing=str(args.ordering),
            )
        )
        model_name = "Hubbard-Holstein"
    else:
        exact_filtered = float(
            exact_ground_energy_sector(
                h_poly,
                num_sites=int(args.L),
                num_particles=num_particles,
                indexing=str(args.ordering),
            )
        )
        model_name = "Hubbard"

    evals = np.linalg.eigvalsh(np.asarray(hmat, dtype=complex))
    exact_full = float(np.real(np.min(evals)))

    ansatz = _build_ansatz(args, num_particles)
    psi_ref = _build_reference_state(args=args, num_particles=num_particles)
    mitigation_cfg = _build_mitigation_config_from_args(args)
    symmetry_mitigation_cfg = _build_symmetry_mitigation_config_from_args(args)
    ideal_symmetry_mitigation_cfg = normalize_ideal_reference_symmetry_mitigation(
        symmetry_mitigation_cfg,
        noise_mode=str(args.noise_mode),
    )

    noisy_cfg = OracleConfig(
        noise_mode=str(args.noise_mode),
        shots=int(args.shots),
        seed=int(args.seed),
        oracle_repeats=int(args.oracle_repeats),
        oracle_aggregate=str(args.oracle_aggregate),
        backend_name=(None if args.backend_name is None else str(args.backend_name)),
        use_fake_backend=bool(args.use_fake_backend),
        allow_aer_fallback=bool(args.allow_aer_fallback),
        aer_fallback_mode="sampler_shots",
        omp_shm_workaround=bool(args.omp_shm_workaround),
        mitigation=dict(mitigation_cfg),
        symmetry_mitigation=dict(symmetry_mitigation_cfg),
    )
    ideal_cfg = OracleConfig(
        noise_mode="ideal",
        shots=int(args.shots),
        seed=int(args.seed),
        oracle_repeats=int(args.oracle_repeats),
        oracle_aggregate=str(args.oracle_aggregate),
        backend_name=None,
        use_fake_backend=False,
        mitigation={"mode": "none", "zne_scales": [], "dd_sequence": None},
        symmetry_mitigation=dict(ideal_symmetry_mitigation_cfg),
    )

    vqe_payload: dict[str, Any] = {
        "success": False,
        "skipped": True,
        "reason": "run_vqe disabled",
    }
    adapt_payload: dict[str, Any] = {
        "success": False,
        "skipped": True,
        "reason": "run_adapt disabled",
    }
    best_theta: np.ndarray | None = None
    selected_adapt_ops: list[AnsatzTerm] | None = None
    selected_adapt_theta: np.ndarray | None = None
    trajectory_rows: list[dict[str, Any]] = []
    legacy_ref: dict[str, Any] | None = None
    legacy_parity_payload: dict[str, Any] = {
        "skipped": True,
        "reference_json": None,
        "observables": list(compare_observables),
        "tolerance": float(args.legacy_parity_tol),
        "time_grid_match": False,
        "reason": "No legacy reference requested.",
        "per_observable": {},
        "passed_all": False,
    }
    if args.legacy_reference_json is not None:
        legacy_ref = _load_legacy_trajectory(Path(args.legacy_reference_json), compare_observables)
        legacy_parity_payload = {
            "skipped": False,
            "reference_json": str(args.legacy_reference_json),
            "observables": list(compare_observables),
            "tolerance": float(args.legacy_parity_tol),
            "time_grid_match": False,
            "reason": "Pending trajectory evaluation.",
            "per_observable": {},
            "passed_all": False,
        }

    with ExpectationOracle(noisy_cfg) as noisy_oracle, ExpectationOracle(ideal_cfg) as ideal_oracle:
        backend_info: NoiseBackendInfo = noisy_oracle.backend_info
        if bool(args.run_adapt):
            adapt_pool = _build_adapt_pool(
                args=args,
                h_poly=h_poly,
                num_particles=num_particles,
            )
            adapt_payload, selected_adapt_ops, selected_adapt_theta = _run_noisy_adapt(
                args=args,
                pool=adapt_pool,
                psi_ref=psi_ref,
                h_qop=h_qop,
                noisy_oracle=noisy_oracle,
                ideal_oracle=ideal_oracle,
            )

        if bool(args.run_vqe):
            vqe_payload, best_theta = _run_noisy_vqe(
                args=args,
                ansatz=ansatz,
                psi_ref=psi_ref,
                h_poly=h_poly,
                h_qop=h_qop,
                noisy_oracle=noisy_oracle,
                ideal_oracle=ideal_oracle,
            )

        if bool(args.run_trotter):
            init_label, init_circuit = _initial_state_selection(
                args=args,
                ansatz=ansatz,
                psi_ref=psi_ref,
                hmat=np.asarray(hmat, dtype=complex),
                best_theta=best_theta,
                adapt_ops=selected_adapt_ops,
                adapt_theta=selected_adapt_theta,
            )
            up0 = _fermion_mode_index(0, "up", str(args.ordering), int(args.L))
            dn0 = _fermion_mode_index(0, "dn", str(args.ordering), int(args.L))
            observables = {
                "energy_static_trotter": h_qop,
                "n_up_site0_trotter": _number_operator_qop(int(h_qop.num_qubits), int(up0)),
                "n_dn_site0_trotter": _number_operator_qop(int(h_qop.num_qubits), int(dn0)),
                "doublon_trotter": _doublon_site_qop(int(h_qop.num_qubits), int(up0), int(dn0)),
            }
            trajectory_rows = _run_noisy_trotter(
                args=args,
                initial_circuit=init_circuit,
                ordered_qop=ordered_qop,
                observables=observables,
                noisy_oracle=noisy_oracle,
                ideal_oracle=ideal_oracle,
            )
        else:
            init_label = str(args.initial_state_source)
    if legacy_ref is not None:
        legacy_parity_payload = _compute_legacy_parity(
            legacy_ref=legacy_ref,
            new_trajectory=trajectory_rows,
            observables=compare_observables,
            tolerance=float(args.legacy_parity_tol),
        )
    delta_uncertainty = _trajectory_delta_uncertainty(trajectory_rows)

    payload: dict[str, Any] = {
        "pipeline": "hh_noise_hardware_validation",
        "model": model_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_command": current_command_string(),
        "settings": {
            "problem": str(args.problem),
            "ansatz": str(args.ansatz),
            "L": int(args.L),
            "t": float(args.t),
            "u": float(args.u),
            "dv": float(args.dv),
            "omega0": float(args.omega0),
            "g_ep": float(args.g_ep),
            "n_ph_max": int(args.n_ph_max),
            "boson_encoding": str(args.boson_encoding),
            "boundary": str(args.boundary),
            "ordering": str(args.ordering),
            "run_vqe": bool(args.run_vqe),
            "run_trotter": bool(args.run_trotter),
            "run_adapt": bool(args.run_adapt),
            "initial_state_source": str(args.initial_state_source),
            "resolved_initial_state_source": str(init_label),
            "t_final": float(args.t_final),
            "num_times": int(args.num_times),
            "suzuki_order": int(args.suzuki_order),
            "trotter_steps": int(args.trotter_steps),
            "exact_steps_multiplier": int(args.exact_steps_multiplier),
            "allow_aer_fallback": bool(args.allow_aer_fallback),
            "omp_shm_workaround": bool(args.omp_shm_workaround),
            "mitigation": str(args.mitigation),
            "symmetry_mitigation_mode": str(args.symmetry_mitigation_mode),
            "zne_scales": (None if args.zne_scales is None else str(args.zne_scales)),
            "dd_sequence": (None if args.dd_sequence is None else str(args.dd_sequence)),
            "mitigation_config": dict(mitigation_cfg),
            "symmetry_mitigation_config": dict(symmetry_mitigation_cfg),
            "vqe_reps": int(args.vqe_reps),
            "vqe_restarts": int(args.vqe_restarts),
            "vqe_maxiter": int(args.vqe_maxiter),
            "vqe_method": str(args.vqe_method),
            "vqe_seed": int(args.vqe_seed),
            "vqe_energy_backend": str(args.vqe_energy_backend),
            "vqe_spsa_a": float(args.vqe_spsa_a),
            "vqe_spsa_c": float(args.vqe_spsa_c),
            "vqe_spsa_alpha": float(args.vqe_spsa_alpha),
            "vqe_spsa_gamma": float(args.vqe_spsa_gamma),
            "vqe_spsa_A": float(args.vqe_spsa_A),
            "vqe_spsa_avg_last": int(args.vqe_spsa_avg_last),
            "vqe_spsa_eval_repeats": int(args.vqe_spsa_eval_repeats),
            "vqe_spsa_eval_agg": str(args.vqe_spsa_eval_agg),
            "adapt_pool": str(args.adapt_pool),
            "adapt_max_depth": int(args.adapt_max_depth),
            "adapt_eps_grad": float(args.adapt_eps_grad),
            "adapt_eps_energy": float(args.adapt_eps_energy),
            "adapt_maxiter": int(args.adapt_maxiter),
            "adapt_seed": int(args.adapt_seed),
            "adapt_inner_optimizer": str(args.adapt_inner_optimizer),
            "adapt_spsa_a": float(args.adapt_spsa_a),
            "adapt_spsa_c": float(args.adapt_spsa_c),
            "adapt_spsa_alpha": float(args.adapt_spsa_alpha),
            "adapt_spsa_gamma": float(args.adapt_spsa_gamma),
            "adapt_spsa_A": float(args.adapt_spsa_A),
            "adapt_spsa_avg_last": int(args.adapt_spsa_avg_last),
            "adapt_spsa_eval_repeats": int(args.adapt_spsa_eval_repeats),
            "adapt_spsa_eval_agg": str(args.adapt_spsa_eval_agg),
            "adapt_allow_repeats": bool(args.adapt_allow_repeats),
            "adapt_gradient_step": float(args.adapt_gradient_step),
            "adapt_min_confidence": float(args.adapt_min_confidence),
            "compare_observables": ",".join(compare_observables),
            "legacy_reference_json": (
                None if args.legacy_reference_json is None else str(args.legacy_reference_json)
            ),
            "legacy_parity_tol": float(args.legacy_parity_tol),
            "output_compare_plot": (
                None if args.output_compare_plot is None else str(args.output_compare_plot)
            ),
        },
        "noise_config": asdict(noisy_cfg),
        "backend": asdict(backend_info),
        "hamiltonian": {
            "num_qubits": int(h_qop.num_qubits),
            "num_terms": int(len(native_order)),
        },
        "ground_state": {
            "exact_energy_full_hilbert": float(exact_full),
            "exact_energy_filtered": float(exact_filtered),
            "filtered_sector": {"n_up": int(num_particles[0]), "n_dn": int(num_particles[1])},
        },
        "vqe": vqe_payload,
        "adapt": adapt_payload,
        "trajectory": trajectory_rows,
        "delta_uncertainty": delta_uncertainty,
        "execution_fallback": {
            "used": bool(backend_info.details.get("fallback_used", False)),
            "mode": backend_info.details.get("fallback_mode"),
            "reason": backend_info.details.get("fallback_reason", ""),
            "aer_failed": bool(backend_info.details.get("aer_failed", False)),
            "mitigation": dict(backend_info.details.get("mitigation", {})),
            "symmetry_mitigation": dict(backend_info.details.get("symmetry_mitigation", {})),
        },
        "legacy_parity": legacy_parity_payload,
    }

    if trajectory_rows:
        final = trajectory_rows[-1]
        energy_metrics = dict(delta_uncertainty.get("energy_static_trotter", {}))
        if not energy_metrics and delta_uncertainty:
            # Fallback: use the first available observable channel.
            first_key = sorted(delta_uncertainty.keys())[0]
            energy_metrics = dict(delta_uncertainty.get(first_key, {}))
        payload["summary"] = {
            "final_energy_delta_noisy_minus_ideal": float(
                final["energy_static_trotter_delta_noisy_minus_ideal"]
            ),
            "final_doublon_delta_noisy_minus_ideal": float(
                final["doublon_trotter_delta_noisy_minus_ideal"]
            ),
            "max_abs_delta": float(energy_metrics.get("max_abs_delta", float("nan"))),
            "max_abs_delta_over_stderr": float(
                energy_metrics.get("max_abs_delta_over_stderr", float("nan"))
            ),
            "mean_abs_delta_over_stderr": float(
                energy_metrics.get("mean_abs_delta_over_stderr", float("nan"))
            ),
        }

    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _ai_log("hh_noise_validation_json_written", path=str(output_json))

    if not bool(args.skip_pdf):
        if not HAS_MATPLOTLIB:
            raise RuntimeError(
                "matplotlib is required for PDF output. Install matplotlib or run with --skip-pdf."
            )
        _write_noise_validation_pdf(pdf_path=output_pdf, payload=payload)
        _ai_log("hh_noise_validation_pdf_written", path=str(output_pdf))

    if args.output_compare_plot is not None:
        _write_legacy_comparison_plot(
            plot_path=Path(args.output_compare_plot),
            payload=payload,
            legacy_ref=legacy_ref,
        )
        _ai_log("hh_noise_validation_compare_plot_written", path=str(args.output_compare_plot))

    _ai_log(
        "hh_noise_validation_done",
        output_json=str(output_json),
        output_pdf=(None if bool(args.skip_pdf) else str(output_pdf)),
    )
    print(f"Wrote JSON: {output_json}")
    if not bool(args.skip_pdf):
        print(f"Wrote PDF:  {output_pdf}")


if __name__ == "__main__":
    main()
