#!/usr/bin/env python3
"""Noisy VQE for the circuit-optimized 7-term HH ansatz on FakeNighthawk.

Runs a shot-based VQE with:
  - FakeNighthawk noise model (full gate + readout errors)
  - M3 (mthree) readout error mitigation
  - SPSA optimizer (noise-robust)
  - Per-Pauli-term measurement with parity expectation

Usage:
    python -m pipelines.hardcoded.hh_noisy_vqe_7term \
        --shots 8192 --maxiter 200 --reps 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 7-term circuit definition (ADAPT ordering — matches CompiledAnsatzExecutor)
# ---------------------------------------------------------------------------

TERMS_7 = [
    ("eeeexy", -0.5),   # uccsd_sing(alpha) — qubits 4,5
    ("eeyxee",  0.5),   # uccsd_sing(beta)  — qubits 2,3
    ("yeeeee",  0.25),  # paop_dbl_p         — qubit 0 (single-qubit)
    ("yezeze",  0.25),  # paop_dbl_p         — qubits 0,2,4
    ("yeyyee", -0.5),   # paop_hopdrag       — qubits 0,2,3
    ("eyeeez", -0.5),   # paop_disp          — qubits 1,5
    ("eyezee", -0.5),   # paop_disp          — qubits 1,3
]

# ADAPT ordering: terms applied in executor's native sequence [0,1,2,3,4,5,6]
TERM_ORDER = list(range(len(TERMS_7)))

EXYZ_TO_IXYZ = {"e": "I", "x": "X", "y": "Y", "z": "Z"}
NQ = 6


def _pauli_rotation_circuit(nq: int, pauli_str: str, param: Parameter) -> QuantumCircuit:
    """Build exp(-i * param/2 * P) for a single Pauli string.

    Maps exyz position i → Qiskit qubit (nq - 1 - i) because exyz position 0
    is MSB in the kron product, while Qiskit qubit 0 is LSB.
    """
    qc = QuantumCircuit(nq)
    active = []
    for i, p in enumerate(pauli_str):
        q = nq - 1 - i  # exyz position → Qiskit qubit
        if p == "x":
            qc.h(q)
            active.append(q)
        elif p == "y":
            qc.sdg(q)
            qc.h(q)
            active.append(q)
        elif p == "z":
            active.append(q)
    if not active:
        return qc
    for j in range(len(active) - 1):
        qc.cx(active[j], active[j + 1])
    qc.rz(param, active[-1])
    for j in range(len(active) - 2, -1, -1):
        qc.cx(active[j], active[j + 1])
    for i, p in enumerate(pauli_str):
        q = nq - 1 - i
        if p == "x":
            qc.h(q)
        elif p == "y":
            qc.h(q)
            qc.s(q)
    return qc


def build_parameterized_circuit() -> tuple[QuantumCircuit, list[Parameter]]:
    """Build the 7-parameter ansatz circuit with ADAPT ordering.

    HF reference state: |000101⟩ (qubits 0 and 2 = |1⟩).
    Each parameter th_i controls exp(-i * th_i/2 * P_i) for the i-th term
    in TERM_ORDER.
    """
    params = [Parameter(f"th{i}") for i in range(7)]
    qc = QuantumCircuit(NQ)
    # Prepare HF reference state: X on qubits 0 and 2
    qc.x(0)
    qc.x(2)
    for idx, term_idx in enumerate(TERM_ORDER):
        pauli_str, _coeff = TERMS_7[term_idx]
        sub = _pauli_rotation_circuit(NQ, pauli_str, params[idx])
        qc.compose(sub, inplace=True)
    return qc, params


def build_hamiltonian_observable() -> SparsePauliOp:
    """Build the HH Hamiltonian as a SparsePauliOp.

    The repo's exyz label convention matches Qiskit's SparsePauliOp kron
    ordering (position 0 = MSB in kron product), so NO string reversal.
    """
    from src.quantum.hubbard_latex_python_pairs import build_hubbard_holstein_hamiltonian
    h_poly = build_hubbard_holstein_hamiltonian(
        dims=2, J=1.0, U=4.0, omega0=1.0, g=0.5, n_ph_max=1,
        boson_encoding="binary", repr_mode="JW", indexing="blocked",
        pbc=False, include_zero_point=True,
    )
    terms = list(h_poly.return_polynomial())
    nq = int(terms[0].nqubit())
    cleaned = []
    for term in terms:
        coeff = complex(term.p_coeff)
        label_exyz = term.pw2strng()
        label_ixyz = "".join(EXYZ_TO_IXYZ[ch] for ch in label_exyz)
        if abs(coeff) > 1e-12:
            cleaned.append((label_ixyz, coeff))
    if not cleaned:
        cleaned = [("I" * nq, 0.0)]
    return SparsePauliOp.from_list(cleaned).simplify(atol=1e-12)


def sv_energy_from_circuit(
    qc: QuantumCircuit, params: list, observable: SparsePauliOp, theta: np.ndarray
) -> float:
    """Evaluate <H> directly from the Qiskit circuit using Statevector.

    This bypasses the CompiledAnsatzExecutor entirely, evaluating the circuit
    in its own parameter space (no coefficient scaling, no reindexing).
    """
    from qiskit.quantum_info import Statevector
    bound = qc.assign_parameters(dict(zip(params, theta.tolist())))
    sv = Statevector.from_instruction(bound)
    return float(np.real(sv.expectation_value(observable)))


# ---------------------------------------------------------------------------
# Noisy energy evaluation
# ---------------------------------------------------------------------------

def _active_qubits_for_label(label_ixyz: str) -> tuple[int, ...]:
    """Return logical qubit indices with non-identity Paulis (Qiskit convention)."""
    n = len(label_ixyz)
    return tuple(q for q in range(n) if label_ixyz[n - 1 - q] != "I")


def _parity_from_counts(counts: dict[str, int], num_bits: int) -> float:
    if num_bits <= 0:
        return 1.0
    shots = sum(counts.values())
    acc = 0.0
    for bitstr, ct in counts.items():
        bitstr = bitstr.replace(" ", "").zfill(num_bits)
        ones = sum(1 for ch in bitstr[-num_bits:] if ch == "1")
        parity = -1.0 if (ones % 2) else 1.0
        acc += parity * ct
    return acc / shots


def _parity_from_quasi(quasi: dict[str, float], num_bits: int) -> float:
    if num_bits <= 0:
        return 1.0
    total = 0.0
    for bitstr, prob in quasi.items():
        bitstr = str(bitstr).replace(" ", "").zfill(num_bits)
        ones = sum(1 for ch in bitstr[-num_bits:] if ch == "1")
        parity = -1.0 if (ones % 2) else 1.0
        total += parity * float(prob)
    return total


class NoisyEnergyEvaluator:
    """Shot-based energy evaluator with M3 readout mitigation."""

    def __init__(
        self,
        compiled_circuit: QuantumCircuit,
        params: list,
        logical_to_physical: list[int],
        observable: SparsePauliOp,
        backend_target: Any,
        shots: int = 8192,
        use_mthree: bool = True,
        seed: int = 7,
    ):
        self.compiled_circuit = compiled_circuit
        self.params = params  # ordered Parameter list for binding
        self.logical_to_physical = logical_to_physical
        self.observable = observable
        self.target = backend_target
        self.shots = shots
        self.use_mthree = use_mthree
        self.seed = seed
        self.nfev = 0

        # M3 setup
        self._mitigator = None
        self._calibrated_qubits: set[tuple[int, ...]] = set()
        if use_mthree:
            import mthree
            self._mthree = mthree
            self._mitigator = mthree.M3Mitigation(backend_target)

    def _ensure_calibration(self, active_physical: tuple[int, ...]) -> None:
        if not self.use_mthree or self._mitigator is None:
            return
        if active_physical in self._calibrated_qubits:
            return
        self._mitigator.cals_from_system(
            qubits=list(active_physical),
            shots=self.shots,
            async_cal=False,
        )
        self._calibrated_qubits.add(active_physical)

    def evaluate(self, theta: np.ndarray, repeat_idx: int = 0) -> float:
        """Evaluate <H> at given parameters with shot noise + optional M3."""
        self.nfev += 1

        # Direct parameter binding: theta[i] → params[i], no coefficient scaling
        param_dict = {p: float(theta[i]) for i, p in enumerate(self.params)}
        bound = self.compiled_circuit.assign_parameters(param_dict)

        total = 0.0
        for label, coeff in self.observable.to_list():
            coeff_c = complex(coeff)
            label_s = str(label).upper()

            # Identity term
            if all(ch == "I" for ch in label_s):
                total += float(np.real(coeff_c))
                continue

            # Find active qubits (Qiskit convention)
            active_logical = _active_qubits_for_label(label_s)
            active_physical = tuple(
                int(self.logical_to_physical[q]) for q in active_logical
            )

            # Build measurement circuit
            meas_qc = bound.copy()
            from qiskit import ClassicalRegister
            creg = ClassicalRegister(len(active_physical), "m")
            meas_qc.add_register(creg)
            for q in active_logical:
                op = label_s[len(label_s) - 1 - q]
                phys = self.logical_to_physical[q]
                if op == "X":
                    meas_qc.h(phys)
                elif op == "Y":
                    meas_qc.sdg(phys)
                    meas_qc.h(phys)
            for idx, phys in enumerate(active_physical):
                meas_qc.measure(phys, creg[idx])

            # Transpile measurement gates into basis set before execution
            meas_qc = transpile(meas_qc, basis_gates=["cz", "rz", "sx", "x", "id", "measure", "reset"],
                                optimization_level=0)

            # Execute
            result = self.target.run(
                meas_qc,
                shots=self.shots,
                seed_simulator=self.seed + repeat_idx + self.nfev,
            ).result()
            counts = result.get_counts()
            if isinstance(counts, list):
                counts = counts[0]

            # M3 or raw parity
            if self.use_mthree and self._mitigator is not None:
                self._ensure_calibration(active_physical)
                quasi = self._mitigator.apply_correction(
                    dict(counts),
                    qubits=list(active_physical),
                )
                exp_val = _parity_from_quasi(dict(quasi), len(active_physical))
            else:
                exp_val = _parity_from_counts(counts, len(active_physical))

            total += float(np.real(coeff_c)) * exp_val

        return total


# ---------------------------------------------------------------------------
# SPSA optimizer
# ---------------------------------------------------------------------------

class SimpleSPSA:
    """Simultaneous Perturbation Stochastic Approximation."""

    def __init__(
        self,
        energy_fn,
        x0: np.ndarray,
        a: float = 0.1,
        c: float = 0.1,
        alpha: float = 0.602,
        gamma: float = 0.101,
        A: float = 10.0,
        maxiter: int = 200,
        callback=None,
    ):
        self.f = energy_fn
        self.x = x0.copy()
        self.a = a
        self.c = c
        self.alpha = alpha
        self.gamma = gamma
        self.A = A
        self.maxiter = maxiter
        self.callback = callback
        self.nfev = 0
        self.history: list[dict] = []

    def step(self, k: int) -> tuple[np.ndarray, float]:
        ak = self.a / (k + 1 + self.A) ** self.alpha
        ck = self.c / (k + 1) ** self.gamma
        delta = 2 * np.random.binomial(1, 0.5, size=len(self.x)) - 1

        x_plus = self.x + ck * delta
        x_minus = self.x - ck * delta

        y_plus = self.f(x_plus)
        y_minus = self.f(x_minus)
        self.nfev += 2

        grad_est = (y_plus - y_minus) / (2 * ck * delta)
        self.x = self.x - ak * grad_est
        return self.x.copy(), 0.5 * (y_plus + y_minus)

    def run(self) -> dict:
        best_x = self.x.copy()
        best_e = float("inf")

        for k in range(self.maxiter):
            x_new, e_est = self.step(k)
            if e_est < best_e:
                best_e = e_est
                best_x = x_new.copy()
            record = {
                "iter": k,
                "energy_est": e_est,
                "best_energy": best_e,
                "nfev": self.nfev,
            }
            self.history.append(record)
            if self.callback:
                self.callback(record)

        return {
            "x": best_x,
            "fun": best_e,
            "nfev": self.nfev,
            "history": self.history,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Noisy VQE for 7-term HH circuit")
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--reps", type=int, default=1, help="Oracle repeats per energy eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-mthree", action="store_true", help="Disable M3 mitigation")
    parser.add_argument("--spsa-a", type=float, default=0.05)
    parser.add_argument("--spsa-c", type=float, default=0.1)
    parser.add_argument("--spsa-A", type=float, default=10.0)
    parser.add_argument("--transpiler-seed", type=int, default=17)
    parser.add_argument("--opt-level", type=int, default=2)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    t0 = time.monotonic()

    print("=" * 60)
    print("Noisy VQE — 7-term HH Circuit on FakeNighthawk")
    print("=" * 60)
    print(f"  Shots:    {args.shots}")
    print(f"  MaxIter:  {args.maxiter}")
    print(f"  M3:       {'ON' if not args.no_mthree else 'OFF'}")
    print(f"  Seed:     {args.seed}")
    print()

    # 1. Build circuit and observable
    print("[1/5] Building circuit and Hamiltonian observable...")
    qc, params = build_parameterized_circuit()
    observable = build_hamiltonian_observable()
    n_terms_h = len(observable.to_list())
    print(f"  Circuit: {qc.num_qubits} qubits, {qc.count_ops()}")
    print(f"  Hamiltonian: {n_terms_h} Pauli terms")

    # Compute sector-filtered exact ground state (correct particle number sector)
    from src.quantum.ed_hubbard_holstein import build_hh_sector_hamiltonian_ed
    h_sector = build_hh_sector_hamiltonian_ed(
        dims=2, J=1.0, U=4.0, omega0=1.0, g=0.5, n_ph_max=1,
        num_particles=(1, 1), indexing="blocked", boson_encoding="binary",
        pbc=False, delta_v=0.0, include_zero_point=True, sparse=False, return_basis=False,
    )
    E_exact = float(np.linalg.eigvalsh(np.asarray(h_sector, dtype=complex))[0])
    print(f"  Sector GS energy: {E_exact:.10f}")

    # Also compute full Hamiltonian GS for reference
    H_mat = observable.to_matrix()
    E_full_gs = float(np.linalg.eigvalsh(H_mat)[0])
    print(f"  Full H GS energy: {E_full_gs:.10f}")

    # 2. Compute known-good starting point from executor
    print("[2/5] Computing statevector starting point from executor...")
    from src.quantum.hubbard_latex_python_pairs import build_hubbard_holstein_hamiltonian
    from src.quantum.hartree_fock_reference_state import hubbard_holstein_reference_state
    from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
    from src.quantum.vqe_latex_python_pairs import expval_pauli_polynomial, half_filled_num_particles
    from pipelines.hardcoded.adapt_circuit_cost import _resolve_scaffold_ops
    from scipy.optimize import minimize as scipy_minimize

    scaffold_path = REPO_ROOT / "artifacts/json/hh_prune_nighthawk_aggressive_5op.json"
    with open(scaffold_path) as f:
        scaffold = json.load(f)

    h_poly = build_hubbard_holstein_hamiltonian(
        dims=2, J=1.0, U=4.0, omega0=1.0, g=0.5, n_ph_max=1,
        boson_encoding="binary", repr_mode="JW", indexing="blocked",
        pbc=False, include_zero_point=True,
    )
    psi_ref = hubbard_holstein_reference_state(
        dims=2, num_particles=tuple(half_filled_num_particles(2)),
        n_ph_max=1, boson_encoding="binary", indexing="blocked",
    )
    ops_resolved = _resolve_scaffold_ops(scaffold, h_poly)
    executor = CompiledAnsatzExecutor(ops_resolved, parameterization_mode="per_pauli_term")

    # Map 7-term indices from executor to TERMS_7
    idx = 0
    all_term_map = []
    for op in ops_resolved:
        poly = op.polynomial
        for term in poly.return_polynomial():
            ps = term.pw2strng()
            coeff = complex(term.p_coeff).real
            all_term_map.append({"idx": idx, "pauli": ps, "coeff": coeff})
            idx += 1

    keep_paulis = {ps for ps, _ in TERMS_7}
    keep_idx = sorted([t["idx"] for t in all_term_map if t["pauli"] in keep_paulis])

    # Optimize the 7-term subspace in the executor
    n_all = executor.runtime_parameter_count
    theta_stored = np.array(scaffold["adapt_vqe"]["optimal_point"])

    def sv_energy_exec(theta):
        psi = executor.prepare_state(theta, psi_ref)
        return float(expval_pauli_polynomial(psi, h_poly))

    def sv_energy_7_exec(x7):
        th = np.zeros(n_all)
        for j, ki in enumerate(keep_idx):
            th[ki] = x7[j]
        return sv_energy_exec(th)

    res_exec = scipy_minimize(sv_energy_7_exec, theta_stored[keep_idx], method="Powell",
                              options={"maxiter": 20000, "ftol": 1e-15})
    E_exec_7 = res_exec.fun
    print(f"  Executor 7-term energy: {E_exec_7:.10f}  |dE|={abs(E_exec_7 - E_exact):.4e}")

    # Convert executor params to circuit params: th_circuit = 2 * dt * coeff
    theta_sv_opt = np.zeros(7)
    for j, ki in enumerate(keep_idx):
        entry = all_term_map[ki]
        theta_sv_opt[j] = 2.0 * res_exec.x[j] * entry["coeff"]

    # Verify: circuit energy at converted params
    E_sv = sv_energy_from_circuit(qc, params, observable, theta_sv_opt)
    print(f"  Circuit SV energy:  {E_sv:.10f}  |dE|={abs(E_sv - E_exact):.4e}")

    # Polish in circuit parameter space
    def sv_obj(theta):
        return sv_energy_from_circuit(qc, params, observable, theta)

    res_polish = scipy_minimize(sv_obj, theta_sv_opt, method="Powell",
                                options={"maxiter": 10000, "ftol": 1e-15})
    if res_polish.fun < E_sv:
        theta_sv_opt = res_polish.x.copy()
        E_sv = res_polish.fun
        print(f"  Polished SV energy: {E_sv:.10f}  |dE|={abs(E_sv - E_exact):.4e}")

    # 3. Transpile for FakeNighthawk
    print("[3/5] Transpiling for FakeNighthawk...")
    from qiskit_ibm_runtime.fake_provider import FakeNighthawk
    backend = FakeNighthawk()
    tc = transpile(
        qc, backend=backend,
        optimization_level=args.opt_level,
        seed_transpiler=args.transpiler_seed,
    )
    ops = dict(tc.count_ops())
    cz_count = ops.get("cz", 0)
    print(f"  Transpiled: {cz_count} CZ, depth {tc.depth()}, size {tc.size()}")
    logical_to_physical = list(tc.layout.final_index_layout(filter_ancillas=True))
    print(f"  Logical→Physical: {logical_to_physical}")

    # 4. Set up noisy AerSimulator (preserves full qubit count for layout)
    print("[4/5] Setting up noisy AerSimulator + evaluator...")
    from qiskit_aer import AerSimulator
    aer_target = AerSimulator.from_backend(backend, seed_simulator=args.seed)
    noise_model = aer_target._options.get("noise_model", None)
    n_noise_q = len(noise_model.noise_qubits) if noise_model else 0
    print(f"  Noise model: {n_noise_q} noisy qubits, {aer_target.num_qubits} total qubits")

    evaluator = NoisyEnergyEvaluator(
        compiled_circuit=tc,
        params=params,
        logical_to_physical=logical_to_physical,
        observable=observable,
        backend_target=aer_target,
        shots=args.shots,
        use_mthree=not args.no_mthree,
        seed=args.seed,
    )

    # Evaluate noisy energy at the SV-optimal point
    print("  Evaluating noisy energy at SV-optimal theta...")
    E_noisy_init = evaluator.evaluate(theta_sv_opt)
    print(f"  Noisy energy at SV-optimal: {E_noisy_init:.6f} (SV: {E_sv:.6f})")

    # 5. Run SPSA
    print(f"[5/5] Running SPSA optimization ({args.maxiter} iterations)...")
    print()

    def callback(record):
        if record["iter"] % 20 == 0 or record["iter"] == args.maxiter - 1:
            print(
                f"  iter {record['iter']:4d}: E_est={record['energy_est']:.6f}  "
                f"best={record['best_energy']:.6f}  nfev={record['nfev']}"
            )

    def noisy_energy_fn(theta):
        """Average over reps for more stable gradient estimates."""
        if args.reps <= 1:
            return evaluator.evaluate(theta)
        vals = [evaluator.evaluate(theta, repeat_idx=r) for r in range(args.reps)]
        return float(np.mean(vals))

    spsa = SimpleSPSA(
        energy_fn=noisy_energy_fn,
        x0=theta_sv_opt,
        a=args.spsa_a,
        c=args.spsa_c,
        A=args.spsa_A,
        maxiter=args.maxiter,
        callback=callback,
    )
    result = spsa.run()

    elapsed = time.monotonic() - t0
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  Best noisy energy: {result['fun']:.6f}")
    print(f"  |dE| from exact:   {abs(result['fun'] - E_exact):.4e}")
    sv_at_best = sv_energy_from_circuit(qc, params, observable, result["x"])
    print(f"  SV energy at best: {sv_at_best:.10f}")
    print(f"  |dE| SV at best:   {abs(sv_at_best - E_exact):.4e}")
    print(f"  Total nfev:        {result['nfev']}")
    print(f"  Elapsed:           {elapsed:.1f}s")
    print(f"  Circuit: {cz_count} CZ, depth {tc.depth()}")
    print(f"  M3 mitigation:     {'ON' if not args.no_mthree else 'OFF'}")

    # Save artifact
    output_path = args.output_json or str(
        REPO_ROOT / "artifacts/json/hh_noisy_vqe_7term_{}.json".format(
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
    )
    artifact = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "hh_noisy_vqe_7term",
        "settings": {
            "shots": args.shots,
            "maxiter": args.maxiter,
            "reps": args.reps,
            "seed": args.seed,
            "mthree": not args.no_mthree,
            "spsa_a": args.spsa_a,
            "spsa_c": args.spsa_c,
            "spsa_A": args.spsa_A,
            "transpiler_seed": args.transpiler_seed,
            "opt_level": args.opt_level,
            "backend": "FakeNighthawk",
        },
        "circuit": {
            "n_qubits": NQ,
            "n_terms": 7,
            "n_params": 7,
            "cz_count": cz_count,
            "depth": tc.depth(),
            "size": tc.size(),
            "layout": logical_to_physical,
            "ordering": "adapt",
        },
        "results": {
            "sector_gs_energy": E_exact,
            "full_hamiltonian_gs_energy": E_full_gs,
            "sv_energy_7term": E_sv,
            "sv_energy_at_noisy_opt": sv_at_best,
            "noisy_best_energy": result["fun"],
            "noisy_abs_delta_e": abs(result["fun"] - E_exact),
            "sv_abs_delta_e_at_noisy_opt": abs(sv_at_best - E_exact),
            "noisy_energy_at_sv_opt": E_noisy_init,
            "optimal_theta": result["x"].tolist(),
            "sv_optimal_theta": theta_sv_opt.tolist(),
            "nfev": result["nfev"],
            "elapsed_s": elapsed,
        },
        "history": result["history"],
    }
    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\n  Artifact saved: {output_path}")


if __name__ == "__main__":
    main()
