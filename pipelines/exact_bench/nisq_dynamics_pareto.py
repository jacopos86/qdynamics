#!/usr/bin/env python3
"""NISQ Dynamics Pareto Explorer — Suzuki-2 Trotter circuits for HH.

Explores the Pareto front of fidelity vs circuit cost for real-time
dynamics under a drive, analogous to the VQE pareto-lean scaffold search.

Knobs explored:
  - Trotter step count (1–10)
  - Term pruning: drop Hamiltonian terms below coefficient threshold
  - Term ordering: native vs sorted-by-weight (commuting terms adjacent)
  - Drive on/off: static vs driven dynamics
  - Propagator: suzuki2

Outputs a JSON artifact with per-variant metrics:
  - Logical CNOT count, single-qubit gate count
  - Transpiled 2Q gate count and depth (FakeMarrakesh / FakeKyiv)
  - Fidelity at selected time points vs exact propagation
  - Energy error |ΔE(t)| at final time

Usage:
  python -m pipelines.exact_bench.nisq_dynamics_pareto [options]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum.hubbard_latex_python_pairs import (
    build_hubbard_holstein_hamiltonian,
    boson_qubits_per_site,
)
from src.quantum.hartree_fock_reference_state import hubbard_holstein_reference_state
from src.quantum.drives_time_potential import build_gaussian_sinusoid_density_drive
from src.quantum.pauli_actions import (
    compile_pauli_action_exyz,
    apply_exp_term,
)
from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ai_log(event: str, **fields: Any) -> None:
    payload = {"event": str(event), "ts_utc": datetime.now(timezone.utc).isoformat(), **fields}
    print(f"AI_LOG {json.dumps(payload, sort_keys=True, default=str)}", flush=True)


def _pauli_weight(label: str) -> int:
    """Number of non-identity Pauli letters in an exyz label."""
    return sum(1 for ch in label if ch != "e")


def _cnot_count_for_label(label: str) -> int:
    """Logical CNOT cost for one exp(-i*angle*P) rotation."""
    w = _pauli_weight(label)
    return 2 * (w - 1) if w > 1 else 0


def _single_qubit_count_for_label(label: str) -> int:
    """Single-qubit gate cost: basis rotations + Rz."""
    w = _pauli_weight(label)
    # Each non-Z active qubit needs 2 basis-change gates (pre + post)
    non_z = sum(1 for ch in label if ch in ("x", "y"))
    basis_changes = 2 * non_z
    # Y needs extra S/Sdg: 2 more per Y qubit
    y_count = sum(1 for ch in label if ch == "y")
    return basis_changes + 2 * y_count + 1  # +1 for the Rz


def _normalize(psi: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(psi)
    return psi / n if n > 0 else psi


# ---------------------------------------------------------------------------
# Hamiltonian construction
# ---------------------------------------------------------------------------

def build_hh_dynamics_hamiltonian(
    L: int,
    t_hop: float,
    U: float,
    omega0: float,
    g: float,
    n_ph_max: int,
    boundary: str,
    ordering: str = "blocked",
) -> dict[str, Any]:
    """Build the HH Hamiltonian and return labels, coefficients, matrix, etc."""
    pbc = boundary == "periodic"
    h_poly = build_hubbard_holstein_hamiltonian(
        dims=L, J=t_hop, U=U, omega0=omega0, g=g, n_ph_max=n_ph_max,
        boson_encoding="binary", repr_mode="JW", indexing=ordering, pbc=pbc,
    )
    nq = h_poly.get_nq()
    poly_terms = h_poly.return_polynomial()

    ordered_labels: list[str] = []
    coeff_map: dict[str, complex] = {}
    for term in poly_terms:
        label = term.pw2strng()
        coeff = complex(term.p_coeff)
        ordered_labels.append(label)
        coeff_map[label] = coeff

    hmat = hamiltonian_matrix(h_poly, tol=0.0)
    evals, evecs = np.linalg.eigh(hmat)

    return {
        "nq": nq,
        "ordered_labels": ordered_labels,
        "coeff_map": coeff_map,
        "hmat": hmat,
        "evals": evals,
        "evecs": evecs,
        "n_sites": L,
    }


# ---------------------------------------------------------------------------
# Suzuki-2 statevector propagation (mirrors hubbard_pipeline logic)
# ---------------------------------------------------------------------------

def evolve_suzuki2(
    psi0: np.ndarray,
    ordered_labels: list[str],
    coeff_map: dict[str, complex],
    compiled_actions: dict[str, Any],
    t_final: float,
    trotter_steps: int,
    drive_provider: Any = None,
    t0: float = 0.0,
) -> np.ndarray:
    """Suzuki-2 time evolution, optionally with time-dependent drive."""
    psi = np.array(psi0, copy=True)
    if abs(t_final) < 1e-15:
        return psi
    dt = float(t_final) / float(trotter_steps)
    half = 0.5 * dt

    for k in range(int(trotter_steps)):
        if drive_provider is not None:
            t_sample = t0 + (float(k) + 0.5) * dt
            drive_map = dict(drive_provider(t_sample))
        else:
            drive_map = {}

        # Forward half-step
        for label in ordered_labels:
            c_total = coeff_map.get(label, 0.0) + complex(drive_map.get(label, 0.0))
            if abs(c_total) < 1e-14:
                continue
            psi = apply_exp_term(psi, compiled_actions[label], c_total, half)
        # Reverse half-step
        for label in reversed(ordered_labels):
            c_total = coeff_map.get(label, 0.0) + complex(drive_map.get(label, 0.0))
            if abs(c_total) < 1e-14:
                continue
            psi = apply_exp_term(psi, compiled_actions[label], c_total, half)

    return _normalize(psi)


def evolve_exact(
    psi0: np.ndarray,
    hmat: np.ndarray,
    t_final: float,
    evals: np.ndarray,
    evecs: np.ndarray,
    drive_provider: Any = None,
    t0: float = 0.0,
    reference_steps: int = 1000,
) -> np.ndarray:
    """Exact time evolution (diag for static, piecewise for driven)."""
    if drive_provider is None:
        evecs_dag = np.conjugate(evecs).T
        psi = evecs @ (np.exp(-1j * evals * t_final) * (evecs_dag @ psi0))
        return _normalize(psi)
    # Piecewise exact for driven case
    psi = np.array(psi0, copy=True)
    nq = int(round(math.log2(len(psi))))
    dt = float(t_final) / float(reference_steps)
    from scipy.linalg import expm
    for k in range(int(reference_steps)):
        t_mid = t0 + (float(k) + 0.5) * dt
        drive_map = dict(drive_provider(t_mid))
        h_total = np.array(hmat, copy=True)
        for label, c_drive in drive_map.items():
            if abs(c_drive) < 1e-15:
                continue
            h_total += complex(c_drive) * _pauli_matrix_exyz(label, nq)
        psi = expm(-1j * h_total * dt) @ psi
    return _normalize(psi)


def _pauli_matrix_exyz(label: str, nq: int) -> np.ndarray:
    """Build full Pauli matrix from exyz label."""
    PAULI = {
        "e": np.array([[1, 0], [0, 1]], dtype=complex),
        "x": np.array([[0, 1], [1, 0]], dtype=complex),
        "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "z": np.array([[1, 0], [0, -1]], dtype=complex),
    }
    mat = np.array([[1.0]], dtype=complex)
    for ch in label:
        mat = np.kron(mat, PAULI[ch])
    return mat


# ---------------------------------------------------------------------------
# Circuit cost analysis (logical)
# ---------------------------------------------------------------------------

def logical_circuit_cost(
    ordered_labels: list[str],
    trotter_steps: int,
) -> dict[str, int]:
    """Compute logical gate counts for a Suzuki-2 Trotter circuit."""
    cnots_per_step = 0
    sq_per_step = 0
    for label in ordered_labels:
        # Each term appears twice per step (forward + reverse half)
        cnots_per_step += 2 * _cnot_count_for_label(label)
        sq_per_step += 2 * _single_qubit_count_for_label(label)

    return {
        "cnots_per_step": cnots_per_step,
        "sq_per_step": sq_per_step,
        "total_cnots": trotter_steps * cnots_per_step,
        "total_sq": trotter_steps * sq_per_step,
        "trotter_steps": trotter_steps,
        "num_terms": len(ordered_labels),
    }


# ---------------------------------------------------------------------------
# Qiskit circuit building for transpilation
# ---------------------------------------------------------------------------

def build_trotter_circuit_qiskit(
    nq: int,
    ordered_labels: list[str],
    coeff_map: dict[str, complex],
    trotter_steps: int,
    t_final: float,
    ref_state: np.ndarray | None = None,
) -> "QuantumCircuit":
    """Build a Qiskit QuantumCircuit for Suzuki-2 Trotter evolution."""
    from qiskit import QuantumCircuit
    from pipelines.hardcoded.adapt_circuit_execution import (
        append_pauli_rotation_exyz,
        append_reference_state,
    )

    qc = QuantumCircuit(int(nq))
    if ref_state is not None:
        append_reference_state(qc, ref_state)

    dt = float(t_final) / float(trotter_steps)
    half = 0.5 * dt

    for _step in range(int(trotter_steps)):
        # Forward half-step
        for label in ordered_labels:
            c = coeff_map.get(label, 0.0)
            if abs(c) < 1e-14:
                continue
            angle = -float(c.real) * half if isinstance(c, complex) and c.imag == 0 else -float(c.real) * half
            # exp(-i * c * half * P) → Rz angle = 2 * c.real * half for real coefficients
            # The standard decomposition: exp(-i*theta/2*P) with angle = 2*c*half
            rotation_angle = 2.0 * float(c.real) * half
            append_pauli_rotation_exyz(qc, label_exyz=label, angle=float(rotation_angle))
        # Reverse half-step
        for label in reversed(ordered_labels):
            c = coeff_map.get(label, 0.0)
            if abs(c) < 1e-14:
                continue
            rotation_angle = 2.0 * float(c.real) * half
            append_pauli_rotation_exyz(qc, label_exyz=label, angle=float(rotation_angle))

    return qc


def transpile_and_measure(
    qc: "QuantumCircuit",
    backend_name: str = "fake_marrakesh",
    optimization_level: int = 2,
    seed: int = 42,
) -> dict[str, Any]:
    """Transpile circuit and return gate count / depth metrics."""
    from qiskit.compiler import transpile
    from pipelines.exact_bench.noise_oracle_runtime import _load_fake_backend
    from pipelines.qiskit_backend_tools import compiled_gate_stats, safe_circuit_depth

    backend, _bname = _load_fake_backend(backend_name)
    qc_compiled = transpile(
        qc, backend=backend,
        optimization_level=optimization_level,
        seed_transpiler=seed,
    )
    stats = compiled_gate_stats(qc_compiled)
    depth = safe_circuit_depth(qc_compiled)

    return {
        "backend": backend_name,
        "optimization_level": optimization_level,
        "depth": depth,
        **stats,
    }


# ---------------------------------------------------------------------------
# Term pruning
# ---------------------------------------------------------------------------

def prune_hamiltonian_terms(
    ordered_labels: list[str],
    coeff_map: dict[str, complex],
    threshold: float,
) -> tuple[list[str], dict[str, complex]]:
    """Drop terms with |coeff| below threshold. Keep identity."""
    pruned_labels = []
    pruned_coeffs = {}
    dropped = 0
    for label in ordered_labels:
        c = coeff_map.get(label, 0.0)
        if abs(c) < threshold and _pauli_weight(label) > 0:
            dropped += 1
            continue
        pruned_labels.append(label)
        pruned_coeffs[label] = c
    return pruned_labels, pruned_coeffs


def reorder_terms_by_weight(
    ordered_labels: list[str],
) -> list[str]:
    """Sort terms by Pauli weight (lighter first) for potential cancellation."""
    return sorted(ordered_labels, key=lambda lbl: (_pauli_weight(lbl), lbl))


def reorder_terms_by_qubit_locality(
    ordered_labels: list[str],
) -> list[str]:
    """Sort terms to cluster operators acting on nearby qubits."""
    def _locality_key(label: str) -> tuple:
        active = [i for i, ch in enumerate(label) if ch != "e"]
        if not active:
            return (0, 0, 0)
        return (min(active), max(active) - min(active), len(active))
    return sorted(ordered_labels, key=_locality_key)


# ---------------------------------------------------------------------------
# Main Pareto sweep
# ---------------------------------------------------------------------------

def run_pareto_sweep(
    L: int = 2,
    t_hop: float = 1.0,
    U: float = 4.0,
    omega0: float = 1.0,
    g: float = 0.5,
    n_ph_max: int = 1,
    boundary: str = "open",
    t_final: float = 5.0,
    num_times: int = 51,
    trotter_steps_list: list[int] | None = None,
    prune_thresholds: list[float] | None = None,
    orderings: list[str] | None = None,
    drive_enabled: bool = True,
    drive_amplitude: float = 0.3,
    drive_omega: float = 2.0,
    drive_tbar: float = 3.0,
    transpile_backends: list[str] | None = None,
    output_json: str | None = None,
) -> dict[str, Any]:
    """Run the full Pareto sweep and return results."""

    if trotter_steps_list is None:
        trotter_steps_list = [1, 2, 3, 4, 5, 7, 10]
    if prune_thresholds is None:
        prune_thresholds = [0.0, 0.05, 0.1, 0.2, 0.3]
    if orderings is None:
        orderings = ["native", "weight_sorted", "qubit_local"]
    if transpile_backends is None:
        transpile_backends = []

    _ai_log("nisq_dynamics_pareto_start", L=L, boundary=boundary,
            t_final=t_final, drive_enabled=drive_enabled,
            trotter_steps=trotter_steps_list, prune_thresholds=prune_thresholds)

    # --- Build Hamiltonian ---
    ham = build_hh_dynamics_hamiltonian(
        L=L, t_hop=t_hop, U=U, omega0=omega0, g=g,
        n_ph_max=n_ph_max, boundary=boundary,
    )
    nq = ham["nq"]
    base_labels = ham["ordered_labels"]
    base_coeffs = ham["coeff_map"]
    hmat = ham["hmat"]
    evals = ham["evals"]
    evecs = ham["evecs"]

    # --- Initial state (HF reference) ---
    psi0 = hubbard_holstein_reference_state(
        dims=L, n_ph_max=n_ph_max, boson_encoding="binary",
        indexing="blocked",
    )

    # --- Drive setup ---
    drive_provider = None
    if drive_enabled:
        drive_obj = build_gaussian_sinusoid_density_drive(
            n_sites=L,
            nq_total=nq,
            indexing="blocked",
            A=drive_amplitude,
            omega=drive_omega,
            tbar=drive_tbar,
            pattern_mode="staggered",
        )
        drive_provider = drive_obj.coeff_map_exyz

    # --- Exact reference trajectory ---
    times = np.linspace(0.0, float(t_final), int(num_times))
    exact_states = []
    exact_energies = []
    _ai_log("computing_exact_reference", num_times=int(num_times))
    for t_val in times:
        psi_exact = evolve_exact(
            psi0, hmat, float(t_val), evals, evecs,
            drive_provider=drive_provider,
            reference_steps=max(2000, 200 * int(t_val + 1)),
        )
        exact_states.append(psi_exact)
        exact_energies.append(float(np.real(np.vdot(psi_exact, hmat @ psi_exact))))
    _ai_log("exact_reference_done")

    # --- Sweep over variants ---
    results: list[dict[str, Any]] = []
    variant_id = 0

    for prune_thresh in prune_thresholds:
        # Prune
        if prune_thresh > 0:
            labels_pruned, coeffs_pruned = prune_hamiltonian_terms(
                base_labels, base_coeffs, prune_thresh,
            )
        else:
            labels_pruned = list(base_labels)
            coeffs_pruned = dict(base_coeffs)

        n_terms = len(labels_pruned)
        if n_terms == 0:
            continue

        for ordering_name in orderings:
            # Reorder
            if ordering_name == "weight_sorted":
                labels_ordered = reorder_terms_by_weight(labels_pruned)
            elif ordering_name == "qubit_local":
                labels_ordered = reorder_terms_by_qubit_locality(labels_pruned)
            else:
                labels_ordered = list(labels_pruned)

            for trotter_steps in trotter_steps_list:
                variant_id += 1
                variant_label = (
                    f"prune={prune_thresh:.2f}_order={ordering_name}_steps={trotter_steps}"
                )

                # Logical cost
                cost = logical_circuit_cost(labels_ordered, trotter_steps)

                # Skip if already too expensive
                if cost["total_cnots"] > 500:
                    results.append({
                        "variant_id": variant_id,
                        "variant_label": variant_label,
                        "prune_threshold": prune_thresh,
                        "ordering": ordering_name,
                        "trotter_steps": trotter_steps,
                        "n_terms": n_terms,
                        "skipped": True,
                        "skip_reason": f"total_cnots={cost['total_cnots']} > 500",
                        **cost,
                    })
                    continue

                # Compile Pauli actions
                compiled = {}
                for label in labels_ordered:
                    compiled[label] = compile_pauli_action_exyz(label, nq)

                # Propagate
                fidelities = []
                energy_errors = []
                t_start = time.perf_counter()

                for idx, t_val in enumerate(times):
                    psi_trot = evolve_suzuki2(
                        psi0, labels_ordered, coeffs_pruned, compiled,
                        float(t_val), trotter_steps,
                        drive_provider=drive_provider,
                    )
                    fid = float(abs(np.vdot(exact_states[idx], psi_trot)) ** 2)
                    e_trot = float(np.real(np.vdot(psi_trot, hmat @ psi_trot)))
                    delta_e = abs(e_trot - exact_energies[idx])
                    fidelities.append(fid)
                    energy_errors.append(delta_e)

                elapsed = time.perf_counter() - t_start

                # Transpile if requested
                transpile_results = {}
                if transpile_backends:
                    qc = build_trotter_circuit_qiskit(
                        nq, labels_ordered, coeffs_pruned,
                        trotter_steps, t_final,
                        ref_state=psi0,
                    )
                    for backend_name in transpile_backends:
                        try:
                            tr = transpile_and_measure(qc, backend_name)
                            transpile_results[backend_name] = tr
                        except Exception as exc:
                            transpile_results[backend_name] = {"error": str(exc)}

                row = {
                    "variant_id": variant_id,
                    "variant_label": variant_label,
                    "prune_threshold": prune_thresh,
                    "ordering": ordering_name,
                    "trotter_steps": trotter_steps,
                    "n_terms": n_terms,
                    "skipped": False,
                    **cost,
                    "fidelity_min": float(min(fidelities)),
                    "fidelity_mean": float(np.mean(fidelities)),
                    "fidelity_final": fidelities[-1],
                    "energy_error_max": float(max(energy_errors)),
                    "energy_error_mean": float(np.mean(energy_errors)),
                    "energy_error_final": energy_errors[-1],
                    "fidelity_trajectory": [round(f, 8) for f in fidelities],
                    "energy_error_trajectory": [round(e, 8) for e in energy_errors],
                    "elapsed_s": elapsed,
                    "transpile": transpile_results,
                }
                results.append(row)

                _ai_log(
                    "nisq_dynamics_variant_done",
                    variant_id=variant_id,
                    variant_label=variant_label,
                    total_cnots=cost["total_cnots"],
                    fidelity_min=row["fidelity_min"],
                    energy_error_final=row["energy_error_final"],
                )

    # --- Build Pareto front ---
    non_skipped = [r for r in results if not r.get("skipped", False)]
    pareto = _compute_pareto_front(non_skipped)

    artifact = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "nisq_dynamics_pareto",
        "settings": {
            "L": L,
            "t_hop": t_hop,
            "U": U,
            "omega0": omega0,
            "g": g,
            "n_ph_max": n_ph_max,
            "boundary": boundary,
            "t_final": t_final,
            "num_times": num_times,
            "drive_enabled": drive_enabled,
            "drive_amplitude": drive_amplitude if drive_enabled else None,
            "drive_omega": drive_omega if drive_enabled else None,
            "drive_tbar": drive_tbar if drive_enabled else None,
            "trotter_steps_list": trotter_steps_list,
            "prune_thresholds": prune_thresholds,
            "orderings": orderings,
            "transpile_backends": transpile_backends,
        },
        "hamiltonian": {
            "nq": nq,
            "n_terms_full": len(base_labels),
            "exact_gs_energy": float(evals[0]),
            "labels": base_labels,
        },
        "results": results,
        "pareto_front": pareto,
        "summary": _build_summary(results, pareto),
    }

    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(artifact, f, indent=2, default=str)
        print(f"Wrote JSON: {out_path}")
        _ai_log("nisq_dynamics_pareto_done", output_json=str(out_path),
                n_variants=len(results), n_pareto=len(pareto))

    return artifact


def _compute_pareto_front(rows: list[dict]) -> list[dict]:
    """Extract Pareto-optimal rows: minimize total_cnots, maximize fidelity_min."""
    if not rows:
        return []
    # Sort by total_cnots ascending
    sorted_rows = sorted(rows, key=lambda r: (r["total_cnots"], -r["fidelity_min"]))
    pareto = []
    best_fidelity = -1.0
    for row in sorted_rows:
        if row["fidelity_min"] > best_fidelity:
            pareto.append({
                "variant_label": row["variant_label"],
                "total_cnots": row["total_cnots"],
                "total_sq": row["total_sq"],
                "n_terms": row["n_terms"],
                "trotter_steps": row["trotter_steps"],
                "fidelity_min": row["fidelity_min"],
                "fidelity_final": row["fidelity_final"],
                "energy_error_final": row["energy_error_final"],
                "transpile": row.get("transpile", {}),
            })
            best_fidelity = row["fidelity_min"]
    return pareto


def _build_summary(results: list[dict], pareto: list[dict]) -> dict:
    """Build summary statistics."""
    non_skipped = [r for r in results if not r.get("skipped", False)]
    nisq_friendly = [r for r in non_skipped if r["total_cnots"] <= 100]
    nisq_best = None
    if nisq_friendly:
        nisq_best = max(nisq_friendly, key=lambda r: r["fidelity_min"])

    return {
        "total_variants": len(results),
        "evaluated_variants": len(non_skipped),
        "skipped_variants": len(results) - len(non_skipped),
        "pareto_size": len(pareto),
        "nisq_friendly_count": len(nisq_friendly),
        "nisq_best": {
            "variant_label": nisq_best["variant_label"],
            "total_cnots": nisq_best["total_cnots"],
            "fidelity_min": nisq_best["fidelity_min"],
            "energy_error_final": nisq_best["energy_error_final"],
        } if nisq_best else None,
        "cheapest_pareto": pareto[0] if pareto else None,
        "best_fidelity_pareto": pareto[-1] if pareto else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="NISQ Dynamics Pareto Explorer")
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--t-hop", type=float, default=1.0)
    p.add_argument("--U", type=float, default=4.0)
    p.add_argument("--omega0", type=float, default=1.0)
    p.add_argument("--g", type=float, default=0.5)
    p.add_argument("--n-ph-max", type=int, default=1)
    p.add_argument("--boundary", choices=["open", "periodic"], default="open")
    p.add_argument("--t-final", type=float, default=5.0)
    p.add_argument("--num-times", type=int, default=51)
    p.add_argument("--trotter-steps", type=str, default="1,2,3,4,5,7,10",
                    help="Comma-separated list of Trotter step counts")
    p.add_argument("--prune-thresholds", type=str, default="0.0,0.05,0.1,0.2,0.3",
                    help="Comma-separated coefficient pruning thresholds")
    p.add_argument("--orderings", type=str, default="native,weight_sorted,qubit_local")
    p.add_argument("--drive", action="store_true", default=True)
    p.add_argument("--no-drive", dest="drive", action="store_false")
    p.add_argument("--drive-amplitude", type=float, default=0.3)
    p.add_argument("--drive-omega", type=float, default=2.0)
    p.add_argument("--drive-tbar", type=float, default=3.0)
    p.add_argument("--transpile-backends", type=str, default="",
                    help="Comma-separated backend names for transpilation (e.g. fake_marrakesh)")
    p.add_argument("--output-json", type=str, default=None)
    args = p.parse_args()

    trotter_steps_list = [int(x) for x in args.trotter_steps.split(",")]
    prune_thresholds = [float(x) for x in args.prune_thresholds.split(",")]
    orderings_list = [x.strip() for x in args.orderings.split(",")]
    transpile_backends = [x.strip() for x in args.transpile_backends.split(",") if x.strip()]

    run_pareto_sweep(
        L=args.L,
        t_hop=args.t_hop,
        U=args.U,
        omega0=args.omega0,
        g=args.g,
        n_ph_max=args.n_ph_max,
        boundary=args.boundary,
        t_final=args.t_final,
        num_times=args.num_times,
        trotter_steps_list=trotter_steps_list,
        prune_thresholds=prune_thresholds,
        orderings=orderings_list,
        drive_enabled=args.drive,
        drive_amplitude=args.drive_amplitude,
        drive_omega=args.drive_omega,
        drive_tbar=args.drive_tbar,
        transpile_backends=transpile_backends,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
