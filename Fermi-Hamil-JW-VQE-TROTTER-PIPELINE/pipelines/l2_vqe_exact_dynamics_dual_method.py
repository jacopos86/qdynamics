#!/usr/bin/env python3
"""L=2 exact-dynamics cross-check for VQE ansatz initial states.

This script compares, for both hardcoded and Qiskit VQE initial states:
1) Python exact evolution via Hamiltonian exponentiation in eigenspace.
2) Qiskit built-in exact evolution via HamiltonianGate.
3) Stored Trotter trajectories from pipeline JSON artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from qiskit import QuantumCircuit
from qiskit.circuit.library import HamiltonianGate
from qiskit.quantum_info import Statevector

ROOT = Path(__file__).resolve().parents[1]

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


def _current_command_string() -> str:
    return " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])


def _normalize_state(psi: np.ndarray) -> np.ndarray:
    nrm = float(np.linalg.norm(psi))
    if nrm <= 0.0:
        raise ValueError("Encountered zero-norm state.")
    return np.asarray(psi, dtype=complex) / nrm


def _pauli_matrix_exyz(label: str) -> np.ndarray:
    out = PAULI_MATS[label[0]]
    for ch in label[1:]:
        out = np.kron(out, PAULI_MATS[ch])
    return out


def _build_hamiltonian_matrix(coeff_entries: list[dict[str, Any]]) -> np.ndarray:
    if not coeff_entries:
        return np.zeros((1, 1), dtype=complex)
    nq = len(str(coeff_entries[0]["label_exyz"]))
    dim = 1 << nq
    hmat = np.zeros((dim, dim), dtype=complex)
    for term in coeff_entries:
        label = str(term["label_exyz"])
        coeff_d = term["coeff"]
        coeff = complex(float(coeff_d["re"]), float(coeff_d["im"]))
        hmat += coeff * _pauli_matrix_exyz(label)
    return hmat


def _state_from_amplitudes_qn_to_q0(amplitudes: dict[str, dict[str, float]], nq: int) -> np.ndarray:
    dim = 1 << nq
    psi = np.zeros(dim, dtype=complex)
    for bitstr, amp in amplitudes.items():
        idx = int(bitstr, 2)
        psi[idx] = complex(float(amp["re"]), float(amp["im"]))
    return _normalize_state(psi)


def _expectation_hamiltonian(psi: np.ndarray, hmat: np.ndarray) -> float:
    return float(np.real(np.vdot(psi, hmat @ psi)))


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


def _evolve_python_exact(psi0: np.ndarray, hmat: np.ndarray, times: np.ndarray, label: str) -> np.ndarray:
    t0 = time.perf_counter()
    _ai_log("exact_python_start", label=label, num_times=int(times.size))
    evals, evecs = np.linalg.eigh(hmat)
    evecs_dag = np.conjugate(evecs).T
    coeff0 = evecs_dag @ psi0

    out = np.zeros((times.size, psi0.size), dtype=complex)
    stride = max(1, int(times.size // 20))
    for idx, tv in enumerate(times):
        out[idx, :] = _normalize_state(evecs @ (np.exp(-1j * evals * float(tv)) * coeff0))
        if idx == 0 or idx == times.size - 1 or ((idx + 1) % stride == 0):
            _ai_log(
                "exact_python_progress",
                label=label,
                step=int(idx + 1),
                total_steps=int(times.size),
                frac=round(float((idx + 1) / times.size), 6),
                elapsed_sec=round(time.perf_counter() - t0, 6),
            )

    _ai_log("exact_python_done", label=label, elapsed_sec=round(time.perf_counter() - t0, 6))
    return out


def _evolve_qiskit_exact(psi0: np.ndarray, hmat: np.ndarray, times: np.ndarray, label: str) -> np.ndarray:
    t0 = time.perf_counter()
    _ai_log("exact_qiskit_start", label=label, num_times=int(times.size))
    nq = int(round(math.log2(psi0.size)))

    out = np.zeros((times.size, psi0.size), dtype=complex)
    stride = max(1, int(times.size // 20))
    for idx, tv in enumerate(times):
        if abs(float(tv)) <= 1e-15:
            psi = np.array(psi0, copy=True)
        else:
            qc = QuantumCircuit(nq)
            qc.append(HamiltonianGate(hmat, time=float(tv)), list(range(nq)))
            psi = np.asarray(Statevector(psi0).evolve(qc).data, dtype=complex)
            psi = _normalize_state(psi)
        out[idx, :] = psi
        if idx == 0 or idx == times.size - 1 or ((idx + 1) % stride == 0):
            _ai_log(
                "exact_qiskit_progress",
                label=label,
                step=int(idx + 1),
                total_steps=int(times.size),
                frac=round(float((idx + 1) / times.size), 6),
                elapsed_sec=round(time.perf_counter() - t0, 6),
            )

    _ai_log("exact_qiskit_done", label=label, elapsed_sec=round(time.perf_counter() - t0, 6))
    return out


def _states_to_observables(states: np.ndarray, hmat: np.ndarray, num_sites: int) -> dict[str, np.ndarray]:
    energy = np.zeros(states.shape[0], dtype=float)
    n_up = np.zeros(states.shape[0], dtype=float)
    n_dn = np.zeros(states.shape[0], dtype=float)
    doublon = np.zeros(states.shape[0], dtype=float)
    for idx, psi in enumerate(states):
        energy[idx] = _expectation_hamiltonian(psi, hmat)
        up, dn = _occupation_site0(psi, num_sites)
        n_up[idx] = up
        n_dn[idx] = dn
        doublon[idx] = _doublon_total(psi, num_sites)
    return {
        "energy": energy,
        "n_up_site0": n_up,
        "n_dn_site0": n_dn,
        "doublon": doublon,
    }


def _arr(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows], dtype=float)


def _max_abs_delta(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def _compute_dataset(payload: dict[str, Any], dataset_label: str) -> dict[str, Any]:
    traj = payload["trajectory"]
    times = _arr(traj, "time")
    num_sites = int(payload["settings"]["L"])
    num_qubits = int(payload["hamiltonian"]["num_qubits"])
    coeff_entries = list(payload["hamiltonian"]["coefficients_exyz"])
    hmat = _build_hamiltonian_matrix(coeff_entries)
    psi0 = _state_from_amplitudes_qn_to_q0(payload["initial_state"]["amplitudes_qn_to_q0"], num_qubits)

    py_states = _evolve_python_exact(psi0, hmat, times, label=f"{dataset_label}:python")
    qk_states = _evolve_qiskit_exact(psi0, hmat, times, label=f"{dataset_label}:qiskit")

    py_obs = _states_to_observables(py_states, hmat, num_sites)
    qk_obs = _states_to_observables(qk_states, hmat, num_sites)

    trotter = {
        "energy": _arr(traj, "energy_trotter"),
        "n_up_site0": _arr(traj, "n_up_site0_trotter"),
        "n_dn_site0": _arr(traj, "n_dn_site0_trotter"),
        "doublon": _arr(traj, "doublon_trotter"),
    }
    payload_exact = {
        "energy": _arr(traj, "energy_exact"),
        "n_up_site0": _arr(traj, "n_up_site0_exact"),
        "n_dn_site0": _arr(traj, "n_dn_site0_exact"),
        "doublon": _arr(traj, "doublon_exact"),
    }

    fidelity_py_qk = np.abs(np.sum(np.conjugate(py_states) * qk_states, axis=1)) ** 2

    summary = {
        "max_abs_delta_python_vs_qiskit_exact": {
            "energy": _max_abs_delta(py_obs["energy"], qk_obs["energy"]),
            "n_up_site0": _max_abs_delta(py_obs["n_up_site0"], qk_obs["n_up_site0"]),
            "n_dn_site0": _max_abs_delta(py_obs["n_dn_site0"], qk_obs["n_dn_site0"]),
            "doublon": _max_abs_delta(py_obs["doublon"], qk_obs["doublon"]),
            "one_minus_fidelity": float(np.max(1.0 - fidelity_py_qk)),
        },
        "max_abs_delta_trotter_vs_python_exact": {
            "energy": _max_abs_delta(trotter["energy"], py_obs["energy"]),
            "n_up_site0": _max_abs_delta(trotter["n_up_site0"], py_obs["n_up_site0"]),
            "n_dn_site0": _max_abs_delta(trotter["n_dn_site0"], py_obs["n_dn_site0"]),
            "doublon": _max_abs_delta(trotter["doublon"], py_obs["doublon"]),
        },
        "max_abs_delta_trotter_vs_qiskit_exact": {
            "energy": _max_abs_delta(trotter["energy"], qk_obs["energy"]),
            "n_up_site0": _max_abs_delta(trotter["n_up_site0"], qk_obs["n_up_site0"]),
            "n_dn_site0": _max_abs_delta(trotter["n_dn_site0"], qk_obs["n_dn_site0"]),
            "doublon": _max_abs_delta(trotter["doublon"], qk_obs["doublon"]),
        },
        "max_abs_delta_payload_exact_vs_python_exact": {
            "energy": _max_abs_delta(payload_exact["energy"], py_obs["energy"]),
            "n_up_site0": _max_abs_delta(payload_exact["n_up_site0"], py_obs["n_up_site0"]),
            "n_dn_site0": _max_abs_delta(payload_exact["n_dn_site0"], py_obs["n_dn_site0"]),
            "doublon": _max_abs_delta(payload_exact["doublon"], py_obs["doublon"]),
        },
        "max_abs_delta_payload_exact_vs_qiskit_exact": {
            "energy": _max_abs_delta(payload_exact["energy"], qk_obs["energy"]),
            "n_up_site0": _max_abs_delta(payload_exact["n_up_site0"], qk_obs["n_up_site0"]),
            "n_dn_site0": _max_abs_delta(payload_exact["n_dn_site0"], qk_obs["n_dn_site0"]),
            "doublon": _max_abs_delta(payload_exact["doublon"], qk_obs["doublon"]),
        },
    }
    _ai_log("dataset_summary", label=dataset_label, summary=summary)

    return {
        "label": dataset_label,
        "num_sites": num_sites,
        "times": times,
        "python_exact": py_obs,
        "qiskit_exact": qk_obs,
        "trotter": trotter,
        "payload_exact": payload_exact,
        "fidelity_python_vs_qiskit_exact": fidelity_py_qk,
        "summary": summary,
    }


def _render_command_page(pdf: PdfPages, command: str) -> None:
    wrapped = textwrap.wrap(command, width=112, subsequent_indent="  ")
    lines = [
        "Executed Command",
        "",
        "Reference: pipelines/PIPELINE_RUN_GUIDE.md",
        "Script: pipelines/l2_vqe_exact_dynamics_dual_method.py",
        "",
        *wrapped,
    ]
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.03, 0.97, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)
    pdf.savefig(fig)
    plt.close(fig)


def _render_dataset_observable_page(pdf: PdfPages, data: dict[str, Any]) -> None:
    times = data["times"]
    py = data["python_exact"]
    qk = data["qiskit_exact"]
    tr = data["trotter"]
    pe = data["payload_exact"]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), sharex=True)
    a00, a01 = axes[0, 0], axes[0, 1]
    a10, a11 = axes[1, 0], axes[1, 1]

    a00.plot(times, py["energy"], label="Python exact exp(H)", color="#111111", linewidth=1.8)
    a00.plot(times, qk["energy"], label="Qiskit built-in exp(H)", color="#2ca02c", linewidth=1.4, linestyle="--")
    a00.plot(times, tr["energy"], label="Trotter", color="#d62728", linewidth=1.2, linestyle=":")
    a00.plot(times, pe["energy"], label="Stored payload exact", color="#1f77b4", linewidth=1.0, alpha=0.7)
    a00.set_title("Energy")
    a00.grid(alpha=0.25)
    a00.legend(fontsize=8)

    a01.plot(times, py["n_up_site0"], label="Python exact", color="#111111", linewidth=1.8)
    a01.plot(times, qk["n_up_site0"], label="Qiskit exact", color="#2ca02c", linewidth=1.4, linestyle="--")
    a01.plot(times, tr["n_up_site0"], label="Trotter", color="#d62728", linewidth=1.2, linestyle=":")
    a01.plot(times, pe["n_up_site0"], label="Stored payload exact", color="#1f77b4", linewidth=1.0, alpha=0.7)
    a01.set_title("Site-0 n_up")
    a01.grid(alpha=0.25)
    a01.legend(fontsize=8)

    a10.plot(times, py["n_dn_site0"], label="Python exact", color="#111111", linewidth=1.8)
    a10.plot(times, qk["n_dn_site0"], label="Qiskit exact", color="#2ca02c", linewidth=1.4, linestyle="--")
    a10.plot(times, tr["n_dn_site0"], label="Trotter", color="#d62728", linewidth=1.2, linestyle=":")
    a10.plot(times, pe["n_dn_site0"], label="Stored payload exact", color="#1f77b4", linewidth=1.0, alpha=0.7)
    a10.set_title("Site-0 n_dn")
    a10.set_xlabel("Time")
    a10.grid(alpha=0.25)
    a10.legend(fontsize=8)

    a11.plot(times, py["doublon"], label="Python exact", color="#111111", linewidth=1.8)
    a11.plot(times, qk["doublon"], label="Qiskit exact", color="#2ca02c", linewidth=1.4, linestyle="--")
    a11.plot(times, tr["doublon"], label="Trotter", color="#d62728", linewidth=1.2, linestyle=":")
    a11.plot(times, pe["doublon"], label="Stored payload exact", color="#1f77b4", linewidth=1.0, alpha=0.7)
    a11.set_title("Total Doublon")
    a11.set_xlabel("Time")
    a11.grid(alpha=0.25)
    a11.legend(fontsize=8)

    fig.suptitle(
        f"L=2 Exact Dynamics Comparison ({data['label']} VQE initial state): Python exp(H) vs Qiskit exp(H) vs Trotter",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def _render_dataset_delta_page(pdf: PdfPages, data: dict[str, Any]) -> None:
    times = data["times"]
    py = data["python_exact"]
    qk = data["qiskit_exact"]
    tr = data["trotter"]
    f = data["fidelity_python_vs_qiskit_exact"]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), sharex=True)
    a00, a01 = axes[0, 0], axes[0, 1]
    a10, a11 = axes[1, 0], axes[1, 1]

    a00.plot(times, np.abs(py["energy"] - qk["energy"]), label="|E_py_exact - E_qiskit_exact|", color="#111111")
    a00.plot(times, np.abs(tr["energy"] - py["energy"]), label="|E_trot - E_py_exact|", color="#d62728")
    a00.plot(times, np.abs(tr["energy"] - qk["energy"]), label="|E_trot - E_qiskit_exact|", color="#2ca02c")
    a00.set_title("Energy deltas")
    a00.grid(alpha=0.25)
    a00.legend(fontsize=8)

    a01.plot(times, np.abs(py["n_up_site0"] - qk["n_up_site0"]), label="|n_up_py - n_up_qk|", color="#111111")
    a01.plot(times, np.abs(tr["n_up_site0"] - py["n_up_site0"]), label="|n_up_trot - n_up_py|", color="#d62728")
    a01.plot(times, np.abs(tr["n_up_site0"] - qk["n_up_site0"]), label="|n_up_trot - n_up_qk|", color="#2ca02c")
    a01.set_title("n_up deltas")
    a01.grid(alpha=0.25)
    a01.legend(fontsize=8)

    a10.plot(times, np.abs(py["n_dn_site0"] - qk["n_dn_site0"]), label="|n_dn_py - n_dn_qk|", color="#111111")
    a10.plot(times, np.abs(tr["n_dn_site0"] - py["n_dn_site0"]), label="|n_dn_trot - n_dn_py|", color="#d62728")
    a10.plot(times, np.abs(tr["n_dn_site0"] - qk["n_dn_site0"]), label="|n_dn_trot - n_dn_qk|", color="#2ca02c")
    a10.set_title("n_dn deltas")
    a10.set_xlabel("Time")
    a10.grid(alpha=0.25)
    a10.legend(fontsize=8)

    a11.plot(times, 1.0 - f, label="1 - Fidelity(py_exact, qiskit_exact)", color="#111111")
    a11.plot(times, np.abs(py["doublon"] - qk["doublon"]), label="|D_py - D_qk|", color="#1f77b4")
    a11.plot(times, np.abs(tr["doublon"] - py["doublon"]), label="|D_trot - D_py|", color="#d62728")
    a11.plot(times, np.abs(tr["doublon"] - qk["doublon"]), label="|D_trot - D_qk|", color="#2ca02c")
    a11.set_title("State/observable deltas")
    a11.set_xlabel("Time")
    a11.grid(alpha=0.25)
    a11.legend(fontsize=8)

    fig.suptitle(f"L=2 Delta Diagnostics ({data['label']} VQE initial state)", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def _render_summary_page(pdf: PdfPages, summary_payload: dict[str, Any]) -> None:
    lines = [
        "L=2 VQE Ansatz Exact-Dynamics Summary",
        "",
        json.dumps(summary_payload, indent=2),
    ]
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8)
    pdf.savefig(fig)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="L=2 VQE ansatz exact dynamics dual-method artifact builder.")
    parser.add_argument(
        "--hardcoded-json",
        type=Path,
        default=ROOT / "artifacts" / "hardcoded_pipeline_L2.json",
    )
    parser.add_argument(
        "--qiskit-json",
        type=Path,
        default=ROOT / "artifacts" / "qiskit_pipeline_L2.json",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=ROOT / "artifacts" / "l2_vqe_ansatz_exact_dynamics_dual_method.pdf",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "artifacts" / "l2_vqe_ansatz_exact_dynamics_dual_method_metrics.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ai_log("l2_exact_dynamics_main_start", settings=vars(args))
    run_command = _current_command_string()

    if not args.hardcoded_json.exists():
        raise FileNotFoundError(f"Missing hardcoded input JSON: {args.hardcoded_json}")
    if not args.qiskit_json.exists():
        raise FileNotFoundError(f"Missing qiskit input JSON: {args.qiskit_json}")

    hardcoded = json.loads(args.hardcoded_json.read_text(encoding="utf-8"))
    qiskit = json.loads(args.qiskit_json.read_text(encoding="utf-8"))

    if int(hardcoded["settings"]["L"]) != 2 or int(qiskit["settings"]["L"]) != 2:
        raise ValueError("This script is scoped to L=2 artifacts only.")

    hardcoded_data = _compute_dataset(hardcoded, "hardcoded")
    qiskit_data = _compute_dataset(qiskit, "qiskit")

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "description": "L=2 exact dynamics of VQE ansatz using Python exact exp(H) and Qiskit exact exp(H), compared to trotterized trajectories.",
        "inputs": {
            "hardcoded_json": str(args.hardcoded_json),
            "qiskit_json": str(args.qiskit_json),
        },
        "datasets": {
            "hardcoded": hardcoded_data["summary"],
            "qiskit": qiskit_data["summary"],
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with PdfPages(str(args.output_pdf)) as pdf:
        _render_command_page(pdf, run_command)
        _render_dataset_observable_page(pdf, hardcoded_data)
        _render_dataset_delta_page(pdf, hardcoded_data)
        _render_dataset_observable_page(pdf, qiskit_data)
        _render_dataset_delta_page(pdf, qiskit_data)
        _render_summary_page(pdf, summary)

    _ai_log(
        "l2_exact_dynamics_main_done",
        output_pdf=str(args.output_pdf),
        output_json=str(args.output_json),
    )
    print(f"Wrote PDF:  {args.output_pdf}")
    print(f"Wrote JSON: {args.output_json}")


if __name__ == "__main__":
    main()
