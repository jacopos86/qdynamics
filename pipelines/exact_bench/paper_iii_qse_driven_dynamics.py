#!/usr/bin/env python3
"""Driven excited-state dynamics matrix on the six Paper-I HH regimes.

Per regime: the production QSE support (geometry selection under the
residual-norm stop, then certified exchange under the Ky Fan objective) is
built on the exact sector ground state, the lowest Ritz root (first
excitation) is taken as the initial state, and it is driven with the
existing staggered-density convention (gaussian-sinusoid envelope
``A sin(wt+phi) exp(-t^2/2 tbar^2)``, T=8, 160 midpoint steps).  The drive
frequency is swept over a grid anchored on QSE-visible transition
frequencies out of the initial root (never on exact/ED gaps).

Two propagation arms start from the identical initial state:

- **frozen QSE**: unitary midpoint propagation in the retained Loewdin
  support with the premeasured projected matrices
  ``M(t) = M0 + c(t) M_drv`` (Eq. frozen_projected_drive of the
  manuscript); measurement-efficient after matrix construction.
- **exact sector**: dense midpoint propagation restricted to the (1,1)
  sector (diagnostic reference only).

Per step the manuscript escape fraction is recorded on the frozen arm:
``rho_esc = [Var_Psi(H(t)) - ||(M(t)-E)y||^2]_+ / (Var_Psi(H(t)) + eps)``
with the variance computed on the represented statevector (diagnostic).

Memory profile: one regime at a time; the largest dense objects are the
three nph7 full-space operators (~16 MB each).  Statevector diagnostics;
never feeds controller decisions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.paper_iii_qse_paper_i_convention_sweep import (
    PAPER_I_REGIMES,
    _build_regime_pool,
    _num_qubits,
)
from pipelines.exact_bench.paper_iii_qse_regime_frontier_sweep import _dense_hamiltonian
from pipelines.excited_dynamics.paper_iii_advisor_demo import (
    _half_filled_sector_indices,
    _lowdin_basis,
    _ritz_states,
)
from pipelines.qse_spectra.compiled_costs import (
    ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
    annotate_basis_with_compiled_costs,
    resolve_cost_weights_preset,
)
from pipelines.qse_spectra.core import QSEBasisVectorPolicy, compute_qse_spectra
from pipelines.qse_spectra.exchange_maintenance import (
    QSEExchangeConfig,
    run_qse_exchange_maintenance,
)
from pipelines.qse_spectra.hh_response_observables import (
    HHResponseLayout,
    build_hh_neutral_response_observable_bundle,
)
from pipelines.qse_spectra.record_selection import (
    StaticRecordSelectionConfig,
    select_static_qse_records,
)
from src.quantum.drives_time_potential import gaussian_sinusoid_waveform
from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "output/diagnostics/paper_iii_driven_dynamics_20260819_v1/driven_dynamics.json"
)
_Q0_POLICY = QSEBasisVectorPolicy(
    reference_projection="q0", basis_vector_normalization="raw_projected"
)
_TARGET_ROOTS = 6
_ESCAPE_EPS = 1.0e-12


def _sector_eigh(dense: np.ndarray, sector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    restricted = dense[np.ix_(sector, sector)]
    return np.linalg.eigh(0.5 * (restricted + restricted.conj().T))


def _midpoint_step(state: np.ndarray, matrix: np.ndarray, dt: float) -> np.ndarray:
    energies, vectors = np.linalg.eigh(0.5 * (matrix + matrix.conj().T))
    return vectors @ (np.exp(-1.0j * float(dt) * energies) * (vectors.conj().T @ state))


def _omega_grid(frequencies: Sequence[float], *, points: int) -> list[float]:
    positive = sorted(f for f in frequencies if f > 1.0e-9)
    if not positive:
        raise ValueError("No positive QSE transition frequencies to anchor the sweep")
    lo, hi = 0.5 * positive[0], 1.1 * positive[-1]
    return [float(x) for x in np.linspace(lo, hi, int(points))]


def _driven_arms(
    *,
    y0: np.ndarray,
    exact0: np.ndarray,
    h0_orth: np.ndarray,
    drive_orth: np.ndarray,
    h0_sector: np.ndarray,
    d_sector: np.ndarray,
    x_sector: np.ndarray,
    phi_ret: np.ndarray,
    h_full: np.ndarray,
    d_full: np.ndarray,
    sector: np.ndarray,
    amplitude: float,
    omega: float,
    tbar: float,
    phi_phase: float,
    t_final: float,
    num_steps: int,
) -> dict[str, Any]:
    dt = float(t_final) / float(num_steps)
    times = np.linspace(0.0, float(t_final), int(num_steps) + 1)
    y = np.asarray(y0, dtype=complex).copy()
    exact = np.asarray(exact0, dtype=complex).copy()

    series: dict[str, list[float]] = {
        "drive_coefficient": [],
        "fidelity_frozen_vs_exact": [],
        "escape_fraction": [],
        "escape_flux": [],
        "hamiltonian_variance": [],
        "represented_drift_sq": [],
        "staggered_density_frozen": [],
        "staggered_density_exact": [],
        "staggered_phonon_frozen": [],
        "staggered_phonon_exact": [],
        "static_energy_frozen": [],
        "static_energy_exact": [],
        "initial_survival_exact": [],
    }
    for step, time_value in enumerate(times):
        coefficient = gaussian_sinusoid_waveform(
            float(time_value), A=float(amplitude), omega=float(omega), tbar=float(tbar), phi=float(phi_phase)
        )
        psi_full = phi_ret @ y
        psi_full = psi_full / float(np.linalg.norm(psi_full))
        psi_sector = psi_full[sector]
        sector_norm = float(np.linalg.norm(psi_sector))
        psi_sector = psi_sector / sector_norm

        m_hat = h0_orth + coefficient * drive_orth
        energy = float(np.real(np.vdot(y, m_hat @ y)))
        h_psi = (h_full @ psi_full) + coefficient * (d_full @ psi_full)
        variance = max(float(np.real(np.vdot(h_psi, h_psi))) - energy * energy, 0.0)
        represented = float(np.linalg.norm(m_hat @ y - energy * y)) ** 2
        unrepresented = max(variance - represented, 0.0)
        escape = unrepresented / (variance + _ESCAPE_EPS)

        series["drive_coefficient"].append(float(coefficient))
        series["fidelity_frozen_vs_exact"].append(float(abs(np.vdot(exact, psi_sector)) ** 2))
        series["escape_fraction"].append(float(escape))
        series["escape_flux"].append(float(np.sqrt(unrepresented)))
        series["hamiltonian_variance"].append(float(variance))
        series["represented_drift_sq"].append(float(represented))
        series["staggered_density_frozen"].append(float(np.real(np.vdot(psi_sector, d_sector @ psi_sector))))
        series["staggered_density_exact"].append(float(np.real(np.vdot(exact, d_sector @ exact))))
        series["staggered_phonon_frozen"].append(float(np.real(np.vdot(psi_sector, x_sector @ psi_sector))))
        series["staggered_phonon_exact"].append(float(np.real(np.vdot(exact, x_sector @ exact))))
        series["static_energy_frozen"].append(float(np.real(np.vdot(psi_sector, h0_sector @ psi_sector))))
        series["static_energy_exact"].append(float(np.real(np.vdot(exact, h0_sector @ exact))))
        series["initial_survival_exact"].append(float(abs(np.vdot(exact0, exact)) ** 2))

        if step == int(num_steps):
            break
        midpoint = gaussian_sinusoid_waveform(
            float(time_value) + 0.5 * dt,
            A=float(amplitude),
            omega=float(omega),
            tbar=float(tbar),
            phi=float(phi_phase),
        )
        y = _midpoint_step(y, h0_orth + midpoint * drive_orth, dt)
        exact = _midpoint_step(exact, h0_sector + midpoint * d_sector, dt)

    fidelity = np.asarray(series["fidelity_frozen_vs_exact"], dtype=float)
    escape_arr = np.asarray(series["escape_fraction"], dtype=float)
    flux = np.asarray(series["escape_flux"], dtype=float)
    density_exact = np.asarray(series["staggered_density_exact"], dtype=float)
    density_err = np.abs(np.asarray(series["staggered_density_frozen"], dtype=float) - density_exact)
    energy_exact = np.asarray(series["static_energy_exact"], dtype=float)
    return {
        "omega": float(omega),
        "series": series,
        "summary": {
            "min_fidelity": float(np.min(fidelity)),
            "final_fidelity": float(fidelity[-1]),
            "max_escape_fraction": float(np.max(escape_arr)),
            "mean_escape_fraction": float(np.mean(escape_arr)),
            "initial_escape_flux": float(flux[0]),
            "max_escape_flux": float(np.max(flux)),
            "escape_flux_amplification": float(np.max(flux) / flux[0]) if flux[0] > 0.0 else None,
            "max_staggered_density_abs_error": float(np.max(density_err)),
            "exact_density_peak_to_peak": float(np.ptp(density_exact)),
            "exact_energy_absorbed": float(energy_exact[-1] - energy_exact[0]),
            "frozen_energy_absorbed": float(
                series["static_energy_frozen"][-1] - series["static_energy_frozen"][0]
            ),
            "min_initial_survival_exact": float(np.min(np.asarray(series["initial_survival_exact"]))),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--t-final", type=float, default=8.0)
    parser.add_argument("--num-steps", type=int, default=160)
    parser.add_argument("--drive-amplitude", type=float, default=0.2)
    parser.add_argument("--drive-tbar", type=float, default=4.0)
    parser.add_argument("--drive-phi", type=float, default=0.0)
    parser.add_argument("--omega-points", type=int, default=13)
    parser.add_argument("--residual-stop", type=float, default=1.0e-3)
    parser.add_argument("--max-rounds", type=int, default=30)
    parser.add_argument("--regimes", default=None, help="Comma-separated regime filter.")
    args = parser.parse_args(argv)

    wanted = None if args.regimes is None else {token.strip() for token in str(args.regimes).split(",")}
    regimes_payload: dict[str, Any] = {}

    for regime, u, g_ep, n_ph_max in PAPER_I_REGIMES:
        if wanted is not None and regime not in wanted:
            continue
        nq = _num_qubits(n_ph_max)
        hamiltonian, basis, _meta = _build_regime_pool(u=u, g_ep=g_ep, n_ph_max=n_ph_max)
        h_full = _dense_hamiltonian(hamiltonian, 1 << nq)
        sector = _half_filled_sector_indices(
            num_sites=2, n_ph_max=n_ph_max, boson_encoding="binary", ordering="blocked", nq_total=nq
        )
        sector_energies, sector_vectors = _sector_eigh(h_full, sector)
        ground = np.zeros(1 << nq, dtype=complex)
        ground[sector] = sector_vectors[:, 0]
        references = [float(x) for x in sector_energies[1 : _TARGET_ROOTS + 1]]

        cost_rows = annotate_basis_with_compiled_costs(
            basis,
            num_qubits=nq,
            oracle_kind=ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
            cost_weights=resolve_cost_weights_preset("two_qubit_only_v1"),
        )
        costs = tuple(row.scalarized_canonical_cost for row in cost_rows)

        print(f"\n== {regime} (u={u}, g={g_ep:.4f}, nph{n_ph_max}) pool={len(basis)}", flush=True)
        selection = select_static_qse_records(
            basis,
            config=StaticRecordSelectionConfig(
                mode="geometry_selected",
                max_records=len(basis),
                geometry_target_roots=_TARGET_ROOTS,
                geometry_cost_discount_alpha=1.0,
                geometry_residual_stop=float(args.residual_stop),
            ),
            hamiltonian=hamiltonian,
            prepared_state=ground,
            basis_vector_policy=_Q0_POLICY,
            compiled_costs=costs,
        )
        exchange = run_qse_exchange_maintenance(
            basis,
            selection.selected_original_indices,
            costs,
            hamiltonian=hamiltonian,
            prepared_state=ground,
            basis_vector_policy=_Q0_POLICY,
            config=QSEExchangeConfig(
                max_rounds=int(args.max_rounds),
                target_root_count=_TARGET_ROOTS,
                insertion_shortlist_size=16,
            ),
        )
        final_indices = list(exchange.final_indices)
        support_2q = float(sum(costs[int(i)] for i in final_indices))
        print(
            f"   support k={len(final_indices)} @{support_2q:.0f}2Q "
            f"(stop: {None if selection.geometry_stop is None else selection.geometry_stop.get('stop_reason')})",
            flush=True,
        )

        qse = compute_qse_spectra(
            hamiltonian,
            ground,
            tuple(basis[int(i)] for i in final_indices),
            basis_vector_policy=_Q0_POLICY,
        )
        energies = np.asarray(qse.eigenvalues, dtype=float).reshape(-1)
        root0_error = abs(float(energies[0]) - references[0])
        ritz = _ritz_states(qse)
        initial_full = ritz[0]
        x_map, phi_ret = _lowdin_basis(qse)

        layout = HHResponseLayout(
            num_sites=2,
            n_ph_max=int(n_ph_max),
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
            total_qubits=nq,
            num_particles=(1, 1),
            source_metadata={"driver": "paper_iii_qse_driven_dynamics", "regime": regime},
        )
        bundle = build_hh_neutral_response_observable_bundle(
            layout=layout, channels=("nn", "XX"), form_factor="staggered", prepared_state=ground
        )
        by_family = {
            str(obs.metadata.get("channel_family")): obs
            for obs in bundle.observables
            if isinstance(obs.metadata, dict) or hasattr(obs.metadata, "get")
        }
        d_full = np.asarray(hamiltonian_matrix(by_family["n"].polynomial), dtype=complex)
        x_full = np.asarray(hamiltonian_matrix(by_family["X"].polynomial), dtype=complex)

        phi_cols = np.column_stack(qse.matrices.basis_matrix_vectors)
        h0_orth = x_map.conj().T @ np.asarray(qse.matrices.hamiltonian, dtype=complex) @ x_map
        drive_orth = x_map.conj().T @ (phi_cols.conj().T @ d_full @ phi_cols) @ x_map
        h0_sector = h_full[np.ix_(sector, sector)]
        d_sector = d_full[np.ix_(sector, sector)]
        x_sector = x_full[np.ix_(sector, sector)]

        y0 = phi_ret.conj().T @ initial_full
        y0 = y0 / float(np.linalg.norm(y0))
        exact0 = initial_full[sector]
        initial_sector_weight = float(np.linalg.norm(exact0) ** 2)
        exact0 = exact0 / float(np.linalg.norm(exact0))

        qse_frequencies = [float(energies[0]) - float(sector_energies[0])] + [
            float(energies[r]) - float(energies[0])
            for r in range(1, min(_TARGET_ROOTS, energies.size))
        ]
        omegas = _omega_grid(qse_frequencies, points=int(args.omega_points))

        sweeps: list[dict[str, Any]] = []
        for omega in omegas:
            record = _driven_arms(
                y0=y0,
                exact0=exact0,
                h0_orth=h0_orth,
                drive_orth=drive_orth,
                h0_sector=h0_sector,
                d_sector=d_sector,
                x_sector=x_sector,
                phi_ret=phi_ret,
                h_full=h_full,
                d_full=d_full,
                sector=sector,
                amplitude=float(args.drive_amplitude),
                omega=float(omega),
                tbar=float(args.drive_tbar),
                phi_phase=float(args.drive_phi),
                t_final=float(args.t_final),
                num_steps=int(args.num_steps),
            )
            sweeps.append(record)
            summary = record["summary"]
            print(
                f"   w={omega:7.4f}  min_fid={summary['min_fidelity']:.6f}  "
                f"esc_flux={summary['initial_escape_flux']:.1e}->{summary['max_escape_flux']:.1e}  "
                f"dens_err={summary['max_staggered_density_abs_error']:.2e}  "
                f"exact_ptp={summary['exact_density_peak_to_peak']:.2e}",
                flush=True,
            )

        regimes_payload[regime] = {
            "u": float(u),
            "g_ep": float(g_ep),
            "n_ph_max": int(n_ph_max),
            "pool_size": len(basis),
            "reference_excitations": references,
            "selection": {
                "support_size": len(final_indices),
                "selected_original_indices": [int(i) for i in final_indices],
                "total_2q": support_2q,
                "retained_rank": int(qse.retained_rank),
                "geometry_stop": None if selection.geometry_stop is None else dict(selection.geometry_stop),
                "exchange_committed_patches": sum(
                    1 for round_record in exchange.rounds if round_record["committed_patch"] is not None
                ),
            },
            "initial_state": {
                "root_index": 0,
                "qse_energy": float(energies[0]),
                "abs_error_vs_reference": float(root0_error),
                "sector_weight": initial_sector_weight,
            },
            "qse_transition_frequencies": [float(f) for f in qse_frequencies],
            "drive": {
                "operator": "staggered_density",
                "waveform": "gaussian_sinusoid A*sin(w t + phi)*exp(-t^2/(2 tbar^2))",
                "amplitude": float(args.drive_amplitude),
                "tbar": float(args.drive_tbar),
                "phi": float(args.drive_phi),
                "t_final": float(args.t_final),
                "num_steps": int(args.num_steps),
                "omega_grid_source": "qse_visible_transition_frequencies_from_root0",
            },
            "times": [float(x) for x in np.linspace(0.0, float(args.t_final), int(args.num_steps) + 1)],
            "omega_sweep": sweeps,
        }
        del h_full, d_full, x_full

    payload = {
        "schema_version": "paper_iii_qse_driven_dynamics_v1",
        "policy": "diagnostic_only_driven_dynamics_matrix",
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "post_run_diagnostic_only": True,
        },
        "target_roots": _TARGET_ROOTS,
        "cost_weights_preset": "two_qubit_only_v1",
        "propagation": "midpoint_magnus2_order2_both_arms_identical_initial_state",
        "escape_definition": "rho_esc = [Var_Psi(H(t)) - ||(M(t)-E)y||^2]_+ / (Var_Psi(H(t)) + 1e-12)",
        "regimes": regimes_payload,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\noutput_json: {args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
