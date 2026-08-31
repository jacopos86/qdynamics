#!/usr/bin/env python3
"""Acceptance tests for CFQM propagation (no Qiskit required)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipelines.hardcoded.hubbard_pipeline as hp
from src.quantum.drives_time_potential import build_gaussian_sinusoid_density_drive
from src.quantum.hubbard_latex_python_pairs import (
    boson_qubits_per_site,
    build_hubbard_hamiltonian,
    build_hubbard_holstein_hamiltonian,
)
from src.quantum.time_propagation.cfqm_propagator import cfqm_step
from src.quantum.time_propagation.cfqm_schemes import get_cfqm_scheme


def _run_cfqm_total(
    *,
    psi0: np.ndarray,
    total_time: float,
    n_steps: int,
    static_coeff_map: dict[str, complex | float],
    drive_coeff_provider,
    ordered_labels: list[str],
    scheme_id: str,
    backend: str,
) -> np.ndarray:
    scheme = get_cfqm_scheme(scheme_id)
    dt = float(total_time) / float(n_steps)
    psi = np.asarray(psi0, dtype=complex)
    cfg = {
        "backend": str(backend),
        "coeff_drop_abs_tol": 0.0,
        "normalize": False,
        "emit_inner_order_warning": False,
    }
    for step in range(int(n_steps)):
        psi = cfqm_step(
            psi=psi,
            t_abs=float(step) * dt,
            dt=dt,
            static_coeff_map=static_coeff_map,
            drive_coeff_provider=drive_coeff_provider,
            ordered_labels=ordered_labels,
            scheme=scheme,
            config=cfg,
        )
    return np.asarray(psi, dtype=complex)


def _estimate_state_order_from_infidelity(
    *,
    errors_inf: list[float],
    k_values: list[int],
    floor: float = 1e-24,
) -> tuple[float, float, tuple[int, int]]:
    """Estimate order from sqrt(infidelity) on smallest-dt resolvable pair."""
    for idx in range(len(errors_inf) - 2, -1, -1):
        e0 = float(errors_inf[idx])
        e1 = float(errors_inf[idx + 1])
        if e0 > floor and e1 > floor:
            p_inf = math.log(e0 / e1) / math.log(2.0)
            p_state = 0.5 * p_inf
            return float(p_state), float(p_inf), (int(k_values[idx]), int(k_values[idx + 1]))
    raise AssertionError("Could not estimate order: all infidelity errors hit numerical floor.")


def test_cfqm4_static_regression_against_exact_expm() -> None:
    """TEST 1: static H(t)=H0 regression, cfqm4 vs one-shot expm reference."""
    ordered_labels = ["x", "z"]
    static_coeff_map = {"x": 0.7, "z": -1.3}
    psi0 = np.array([0.3 + 0.1j, -0.2 + 0.9j], dtype=complex)
    psi0 = psi0 / np.linalg.norm(psi0)
    total_time = 1.75
    n_steps = 23

    psi_cfqm = _run_cfqm_total(
        psi0=psi0,
        total_time=total_time,
        n_steps=n_steps,
        static_coeff_map=static_coeff_map,
        drive_coeff_provider=None,
        ordered_labels=ordered_labels,
        scheme_id="cfqm4",
        backend="dense_expm",
    )

    h0 = hp._build_hamiltonian_matrix({k: complex(v) for k, v in static_coeff_map.items()})
    psi_ref = expm((-1j * float(total_time)) * h0) @ psi0
    err_l2 = float(np.linalg.norm(psi_cfqm - psi_ref))
    assert err_l2 <= 1e-10, f"Static regression failed: ||psi-psi_ref||_2={err_l2:.3e}"


def test_cfqm4_a0_safe_test_invariance_via_real_drive_provider() -> None:
    """TEST 2: A=0 invariance using the actual onsite-density drive provider path."""
    h_poly = build_hubbard_hamiltonian(
        dims=2,
        t=1.0,
        U=4.0,
        v=0.0,
        repr_mode="JW",
        indexing="blocked",
        pbc=True,
    )
    native_order, coeff_map = hp._collect_hardcoded_terms_exyz(h_poly)
    ordered_labels = list(native_order)

    drive_a0 = build_gaussian_sinusoid_density_drive(
        n_sites=2,
        nq_total=4,
        indexing="blocked",
        A=0.0,
        omega=1.1,
        tbar=2.0,
        phi=0.3,
        pattern_mode="staggered",
        include_identity=False,
        coeff_tol=0.0,
    )
    drive_labels = set(drive_a0.template.labels_exyz(include_identity=False))
    missing = sorted(drive_labels.difference(ordered_labels))
    ordered_labels = list(ordered_labels) + list(missing)

    rng = np.random.default_rng(7)
    psi0 = rng.normal(size=16) + 1j * rng.normal(size=16)
    psi0 = np.asarray(psi0 / np.linalg.norm(psi0), dtype=complex)

    total_time = 2.0
    n_steps = 40
    psi_none = _run_cfqm_total(
        psi0=psi0,
        total_time=total_time,
        n_steps=n_steps,
        static_coeff_map=coeff_map,
        drive_coeff_provider=None,
        ordered_labels=ordered_labels,
        scheme_id="cfqm4",
        backend="dense_expm",
    )
    psi_a0 = _run_cfqm_total(
        psi0=psi0,
        total_time=total_time,
        n_steps=n_steps,
        static_coeff_map=coeff_map,
        drive_coeff_provider=drive_a0.coeff_map_exyz,
        ordered_labels=ordered_labels,
        scheme_id="cfqm4",
        backend="dense_expm",
    )
    err_l2 = float(np.linalg.norm(psi_none - psi_a0))
    assert err_l2 <= 1e-10, f"A=0 safe-test failed: ||psi_none-psi_A0||_2={err_l2:.3e}"


def test_manufactured_one_qubit_order_confirmation() -> None:
    """TEST 3: manufactured noncommuting one-qubit order check.

    Split choice:
    - Option A is used.
    - static_coeff_map = {'x': lam}
    - drive returns y/z only (time dependent).
    """
    lam = 6.0
    total_time = 1.0
    k_values = [3, 4, 5, 6, 7]
    ordered_labels = ["x", "y", "z"]
    psi0 = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex)

    # NOTE: This sign convention is the Schrödinger-consistent partner of the
    # provided U_exact under dpsi/dt = -i H(t) psi.
    def drive_provider(t_abs: float) -> dict[str, float]:
        t = float(t_abs)
        return {
            "y": -2.0 * lam * t * math.sin(2.0 * lam * t),
            "z": 2.0 * lam * t * math.cos(2.0 * lam * t),
        }

    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    psi_ref = (expm(-1j * lam * total_time * X) @ expm(-1j * lam * (total_time**2) * Z)) @ psi0

    def run_errors(scheme_id: str, backend: str) -> list[float]:
        out: list[float] = []
        for k in k_values:
            n_steps = 2**k
            psi_k = _run_cfqm_total(
                psi0=psi0,
                total_time=total_time,
                n_steps=n_steps,
                static_coeff_map={"x": lam},
                drive_coeff_provider=drive_provider,
                ordered_labels=ordered_labels,
                scheme_id=scheme_id,
                backend=backend,
            )
            fid = float(abs(np.vdot(psi_ref, psi_k)) ** 2)
            out.append(max(1.0 - fid, 1e-30))
        return out

    err_cfqm4 = run_errors("cfqm4", "dense_expm")
    err_cfqm6 = run_errors("cfqm6", "dense_expm")
    err_suz2 = run_errors("cfqm4", "pauli_suzuki2")

    p4_state, p4_inf, pair4 = _estimate_state_order_from_infidelity(errors_inf=err_cfqm4, k_values=k_values)
    p6_state, p6_inf, pair6 = _estimate_state_order_from_infidelity(errors_inf=err_cfqm6, k_values=k_values)
    p2_state, p2_inf, pair2 = _estimate_state_order_from_infidelity(errors_inf=err_suz2, k_values=k_values)

    print(
        "TEST3 slopes (state-order from infidelity): "
        f"cfqm4={p4_state:.6f} (inf-slope={p4_inf:.6f}, pair={pair4}), "
        f"cfqm6={p6_state:.6f} (inf-slope={p6_inf:.6f}, pair={pair6}), "
        f"pauli_suzuki2={p2_state:.6f} (inf-slope={p2_inf:.6f}, pair={pair2})"
    )

    assert 3.7 <= p4_state <= 4.7, f"cfqm4 state-order slope out of range: {p4_state:.6f}"
    assert 5.3 <= p6_state <= 6.9, f"cfqm6 state-order slope out of range: {p6_state:.6f}"
    assert 1.7 <= p2_state <= 2.3, f"pauli_suzuki2 state-order slope out of range: {p2_state:.6f}"


def test_small_hh_cfqm4_converges_toward_fine_piecewise_exact() -> None:
    """TEST 4: small HH sanity vs fine piecewise_exact reference."""
    L = 2
    n_ph_max = 1
    h_poly = build_hubbard_holstein_hamiltonian(
        dims=L,
        J=1.0,
        U=4.0,
        omega0=1.0,
        g=0.5,
        n_ph_max=n_ph_max,
        boson_encoding="binary",
        repr_mode="JW",
        indexing="blocked",
        pbc=True,
    )
    native_order, coeff_map = hp._collect_hardcoded_terms_exyz(h_poly)
    ordered_labels = list(native_order)
    qpb = int(boson_qubits_per_site(n_ph_max, "binary"))
    nq_total = 2 * L + L * qpb

    drive = build_gaussian_sinusoid_density_drive(
        n_sites=L,
        nq_total=nq_total,
        indexing="blocked",
        A=2.0,
        omega=4.0,
        tbar=0.5,
        phi=1.1,
        pattern_mode="staggered",
        include_identity=False,
        coeff_tol=0.0,
    )
    drive_labels = set(drive.template.labels_exyz(include_identity=False))
    ordered_labels = list(ordered_labels) + sorted(drive_labels.difference(ordered_labels))

    hmat_static = hp._build_hamiltonian_matrix(coeff_map)
    psi0 = np.zeros(hmat_static.shape[0], dtype=complex)
    psi0[1] = 1.0 + 0.0j

    total_time = 10.0
    fine_steps = 4096
    psi_ref = hp._evolve_piecewise_exact(
        psi0=psi0,
        hmat_static=hmat_static,
        drive_coeff_provider_exyz=drive.coeff_map_exyz,
        time_value=total_time,
        trotter_steps=fine_steps,
        t0=0.0,
        time_sampling="midpoint",
    )

    coarse_steps = [8, 16, 32]
    errs: list[float] = []
    for n_steps in coarse_steps:
        psi_cfqm = _run_cfqm_total(
            psi0=psi0,
            total_time=total_time,
            n_steps=n_steps,
            static_coeff_map=coeff_map,
            drive_coeff_provider=drive.coeff_map_exyz,
            ordered_labels=ordered_labels,
            scheme_id="cfqm4",
            backend="expm_multiply_sparse",
        )
        errs.append(float(np.linalg.norm(psi_cfqm - psi_ref)))

    print(
        "TEST4 HH errors vs fine piecewise_exact: "
        f"n={coarse_steps[0]} -> {errs[0]:.6e}, "
        f"n={coarse_steps[1]} -> {errs[1]:.6e}, "
        f"n={coarse_steps[2]} -> {errs[2]:.6e}"
    )

    assert errs[1] < errs[0], "HH sanity failed: error did not decrease from n=8 to n=16."
    assert errs[2] < errs[1], "HH sanity failed: error did not decrease from n=16 to n=32."

