#!/usr/bin/env python3
"""Adaptive-QSE driven propagation at the worst-escape drive frequency.

Consumes the frozen driven-dynamics sweep
(``paper_iii_qse_driven_dynamics.py`` output), picks per regime the drive
frequency with the largest frozen-arm escape flux (a controller-visible
criterion: escape flux is measurement-compatible), and repropagates with
**adaptive support growth**: at every step where the escape flux exceeds a
threshold (and outside a short cooldown), the pool candidate whose
manifold-orthogonal component best aligns with the unrepresented drift
vector is admitted, the retained Loewdin support is rebuilt, and the
current state is re-injected into the grown manifold.  The frozen arm at
the same frequency is rerun for a like-for-like comparison.

Growth uses only quantities that are premeasurable in the QSE workflow
(candidate/manifold overlaps and drift alignment on the represented
state); exact-sector propagation remains a diagnostic reference only.

Memory profile: one regime at a time; candidate images are statevectors
(pool x dim <= 200 x 1024 complex ~ 3 MB).  Statevector diagnostics; never
feeds controller decisions.
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
from pipelines.qse_spectra.core import (
    QSEBasisVectorPolicy,
    QSEPruningConfig,
    _apply_basis_element,
    compute_qse_spectra,
)
from pipelines.qse_spectra.hh_response_observables import (
    HHResponseLayout,
    build_hh_neutral_response_observable_bundle,
)
from src.quantum.drives_time_potential import gaussian_sinusoid_waveform
from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix

DEFAULT_SWEEP = (
    REPO_ROOT
    / "output/diagnostics/paper_iii_driven_dynamics_20260819_v1/driven_dynamics.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "output/diagnostics/paper_iii_driven_dynamics_20260819_v1/adaptive_dynamics.json"
)
_Q0_POLICY = QSEBasisVectorPolicy(
    reference_projection="q0", basis_vector_normalization="raw_projected"
)
_TARGET_ROOTS = 6
_NOVELTY_FLOOR = 1.0e-6


def _midpoint_step(state: np.ndarray, matrix: np.ndarray, dt: float) -> np.ndarray:
    energies, vectors = np.linalg.eigh(0.5 * (matrix + matrix.conj().T))
    return vectors @ (np.exp(-1.0j * float(dt) * energies) * (vectors.conj().T @ state))


def _pool_images(basis: Sequence[Any], ground: np.ndarray, *, nq: int) -> list[np.ndarray]:
    cfg = QSEPruningConfig()
    cache: dict[str, Any] = {}
    psi = np.asarray(ground, dtype=complex).reshape(-1)
    images: list[np.ndarray] = []
    for element in basis:
        raw = np.asarray(
            _apply_basis_element(element, psi, nq=int(nq), config=cfg, pauli_action_cache=cache),
            dtype=complex,
        ).reshape(-1)
        images.append(raw - complex(np.vdot(psi, raw)) * psi)
    return images


def _manifold(
    hamiltonian: Any,
    ground: np.ndarray,
    basis: Sequence[Any],
    support: Sequence[int],
    d_full: np.ndarray,
) -> dict[str, Any]:
    qse = compute_qse_spectra(
        hamiltonian,
        ground,
        tuple(basis[int(i)] for i in support),
        basis_vector_policy=_Q0_POLICY,
    )
    x_map, phi_ret = _lowdin_basis(qse)
    phi_cols = np.column_stack(qse.matrices.basis_matrix_vectors)
    h0_orth = x_map.conj().T @ np.asarray(qse.matrices.hamiltonian, dtype=complex) @ x_map
    drive_orth = x_map.conj().T @ (phi_cols.conj().T @ d_full @ phi_cols) @ x_map
    return {"qse": qse, "phi_ret": phi_ret, "h0_orth": h0_orth, "drive_orth": drive_orth}


def _propagate(
    *,
    adaptive: bool,
    y0: np.ndarray,
    exact0: np.ndarray,
    manifold: dict[str, Any],
    hamiltonian: Any,
    ground: np.ndarray,
    basis: Sequence[Any],
    support: list[int],
    pool_images: Sequence[np.ndarray] | None,
    costs: Sequence[float],
    h0_sector: np.ndarray,
    d_sector: np.ndarray,
    h_full: np.ndarray,
    d_full: np.ndarray,
    sector: np.ndarray,
    amplitude: float,
    omega: float,
    tbar: float,
    phi_phase: float,
    t_final: float,
    num_steps: int,
    growth_flux_threshold: float,
    max_additions: int,
    cooldown_steps: int,
) -> dict[str, Any]:
    dt = float(t_final) / float(num_steps)
    times = np.linspace(0.0, float(t_final), int(num_steps) + 1)
    support = list(support)
    phi_ret = manifold["phi_ret"]
    h0_orth = manifold["h0_orth"]
    drive_orth = manifold["drive_orth"]
    y = np.asarray(y0, dtype=complex).copy()
    exact = np.asarray(exact0, dtype=complex).copy()

    fidelity: list[float] = []
    flux_series: list[float] = []
    density_frozen: list[float] = []
    density_exact_series: list[float] = []
    growth_events: list[dict[str, Any]] = []
    last_growth_step = -(10**9)
    injection_losses: list[float] = []

    for step, time_value in enumerate(times):
        coefficient = gaussian_sinusoid_waveform(
            float(time_value), A=float(amplitude), omega=float(omega), tbar=float(tbar), phi=float(phi_phase)
        )
        psi_full = phi_ret @ y
        psi_full = psi_full / float(np.linalg.norm(psi_full))
        psi_sector = psi_full[sector]
        psi_sector = psi_sector / float(np.linalg.norm(psi_sector))

        m_hat = h0_orth + coefficient * drive_orth
        energy = float(np.real(np.vdot(y, m_hat @ y)))
        h_psi = (h_full @ psi_full) + coefficient * (d_full @ psi_full)
        variance = max(float(np.real(np.vdot(h_psi, h_psi))) - energy * energy, 0.0)
        represented = float(np.linalg.norm(m_hat @ y - energy * y)) ** 2
        flux = float(np.sqrt(max(variance - represented, 0.0)))

        if (
            adaptive
            and pool_images is not None
            and flux > float(growth_flux_threshold)
            and len(growth_events) < int(max_additions)
            and (step - last_growth_step) >= int(cooldown_steps)
        ):
            residual = h_psi - energy * psi_full
            residual_perp = residual - phi_ret @ (phi_ret.conj().T @ residual)
            residual_perp_norm = float(np.linalg.norm(residual_perp))
            best_index, best_score = None, 0.0
            if residual_perp_norm > 0.0:
                selected = set(int(i) for i in support)
                for index, image in enumerate(pool_images):
                    if index in selected:
                        continue
                    image_norm = float(np.linalg.norm(image))
                    # Absolute floor: records whose q0-projected image is
                    # numerically zero (e.g. identity) have noise-scale
                    # directions that can align spuriously.
                    if image_norm <= 1.0e-10:
                        continue
                    perp = image - phi_ret @ (phi_ret.conj().T @ image)
                    perp_norm = float(np.linalg.norm(perp))
                    if perp_norm / image_norm < _NOVELTY_FLOOR:
                        continue
                    score = abs(complex(np.vdot(perp, residual_perp))) / (perp_norm * residual_perp_norm)
                    if score > best_score:
                        best_index, best_score = int(index), float(score)
            if best_index is not None:
                support.append(best_index)
                rebuilt = _manifold(hamiltonian, ground, basis, support, d_full)
                phi_ret = rebuilt["phi_ret"]
                h0_orth = rebuilt["h0_orth"]
                drive_orth = rebuilt["drive_orth"]
                y_new = phi_ret.conj().T @ psi_full
                injection_norm = float(np.linalg.norm(y_new))
                injection_losses.append(max(1.0 - injection_norm**2, 0.0))
                y = y_new / injection_norm
                last_growth_step = step
                growth_events.append(
                    {
                        "step_index": int(step),
                        "time": float(time_value),
                        "added_original_index": int(best_index),
                        "added_name": str(basis[int(best_index)].name),
                        "alignment_score": float(best_score),
                        "escape_flux_before": float(flux),
                        "added_2q": float(costs[int(best_index)]),
                        "support_size_after": len(support),
                        "state_injection_loss": float(injection_losses[-1]),
                    }
                )
                psi_full = phi_ret @ y
                psi_full = psi_full / float(np.linalg.norm(psi_full))
                psi_sector = psi_full[sector]
                psi_sector = psi_sector / float(np.linalg.norm(psi_sector))
                m_hat = h0_orth + coefficient * drive_orth
                energy = float(np.real(np.vdot(y, m_hat @ y)))
                h_psi = (h_full @ psi_full) + coefficient * (d_full @ psi_full)
                variance = max(float(np.real(np.vdot(h_psi, h_psi))) - energy * energy, 0.0)
                represented = float(np.linalg.norm(m_hat @ y - energy * y)) ** 2
                flux = float(np.sqrt(max(variance - represented, 0.0)))

        fidelity.append(float(abs(np.vdot(exact, psi_sector)) ** 2))
        flux_series.append(float(flux))
        density_frozen.append(float(np.real(np.vdot(psi_sector, d_sector @ psi_sector))))
        density_exact_series.append(float(np.real(np.vdot(exact, d_sector @ exact))))

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

    fid = np.asarray(fidelity, dtype=float)
    flux_arr = np.asarray(flux_series, dtype=float)
    dens_err = np.abs(np.asarray(density_frozen) - np.asarray(density_exact_series))
    return {
        "arm": "adaptive" if adaptive else "frozen",
        "series": {
            "fidelity_vs_exact": fidelity,
            "escape_flux": flux_series,
            "staggered_density": density_frozen,
            "staggered_density_exact": density_exact_series,
        },
        "growth_events": growth_events,
        "final_support_size": len(support),
        "final_support_indices": [int(i) for i in support],
        "added_2q_total": float(sum(event["added_2q"] for event in growth_events)),
        "summary": {
            "min_fidelity": float(np.min(fid)),
            "final_fidelity": float(fid[-1]),
            "max_escape_flux": float(np.max(flux_arr)),
            "final_escape_flux": float(flux_arr[-1]),
            "max_staggered_density_abs_error": float(np.max(dens_err)),
            "growth_event_count": len(growth_events),
            "max_state_injection_loss": float(max(injection_losses, default=0.0)),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-json", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--growth-flux-threshold", type=float, default=1.0e-2)
    parser.add_argument("--max-additions", type=int, default=16)
    parser.add_argument("--cooldown-steps", type=int, default=5)
    parser.add_argument("--regimes", default=None, help="Comma-separated regime filter.")
    args = parser.parse_args(argv)

    sweep = json.loads(Path(args.sweep_json).read_text(encoding="utf-8"))
    wanted = None if args.regimes is None else {token.strip() for token in str(args.regimes).split(",")}
    regimes_payload: dict[str, Any] = {}

    for regime, u, g_ep, n_ph_max in PAPER_I_REGIMES:
        if wanted is not None and regime not in wanted:
            continue
        if regime not in sweep["regimes"]:
            continue
        record = sweep["regimes"][regime]
        worst = max(record["omega_sweep"], key=lambda row: float(row["summary"]["max_escape_flux"]))
        omega = float(worst["omega"])
        drive = record["drive"]
        support = [int(i) for i in record["selection"]["selected_original_indices"]]

        nq = _num_qubits(n_ph_max)
        hamiltonian, basis, _meta = _build_regime_pool(u=u, g_ep=g_ep, n_ph_max=n_ph_max)
        h_full = _dense_hamiltonian(hamiltonian, 1 << nq)
        sector = _half_filled_sector_indices(
            num_sites=2, n_ph_max=n_ph_max, boson_encoding="binary", ordering="blocked", nq_total=nq
        )
        restricted = h_full[np.ix_(sector, sector)]
        sector_energies, sector_vectors = np.linalg.eigh(0.5 * (restricted + restricted.conj().T))
        ground = np.zeros(1 << nq, dtype=complex)
        ground[sector] = sector_vectors[:, 0]

        cost_rows = annotate_basis_with_compiled_costs(
            basis,
            num_qubits=nq,
            oracle_kind=ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
            cost_weights=resolve_cost_weights_preset("two_qubit_only_v1"),
        )
        costs = tuple(row.scalarized_canonical_cost for row in cost_rows)

        layout = HHResponseLayout(
            num_sites=2,
            n_ph_max=int(n_ph_max),
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
            total_qubits=nq,
            num_particles=(1, 1),
            source_metadata={"driver": "paper_iii_qse_adaptive_dynamics", "regime": regime},
        )
        bundle = build_hh_neutral_response_observable_bundle(
            layout=layout, channels=("nn",), form_factor="staggered", prepared_state=ground
        )
        d_full = np.asarray(hamiltonian_matrix(bundle.observables[0].polynomial), dtype=complex)
        h0_sector = h_full[np.ix_(sector, sector)]
        d_sector = d_full[np.ix_(sector, sector)]

        manifold = _manifold(hamiltonian, ground, basis, support, d_full)
        ritz = _ritz_states(manifold["qse"])
        initial_full = ritz[0]
        y0 = manifold["phi_ret"].conj().T @ initial_full
        y0 = y0 / float(np.linalg.norm(y0))
        exact0 = initial_full[sector]
        exact0 = exact0 / float(np.linalg.norm(exact0))
        pool_images = _pool_images(basis, ground, nq=nq)

        print(
            f"\n== {regime} (u={u}, g={g_ep:.4f}, nph{n_ph_max}) worst omega={omega:.4f} "
            f"support k={len(support)}",
            flush=True,
        )
        arms: dict[str, Any] = {}
        for adaptive in (False, True):
            result = _propagate(
                adaptive=adaptive,
                y0=y0,
                exact0=exact0,
                manifold=manifold,
                hamiltonian=hamiltonian,
                ground=ground,
                basis=basis,
                support=list(support),
                pool_images=pool_images if adaptive else None,
                costs=costs,
                h0_sector=h0_sector,
                d_sector=d_sector,
                h_full=h_full,
                d_full=d_full,
                sector=sector,
                amplitude=float(drive["amplitude"]),
                omega=omega,
                tbar=float(drive["tbar"]),
                phi_phase=float(drive["phi"]),
                t_final=float(drive["t_final"]),
                num_steps=int(drive["num_steps"]),
                growth_flux_threshold=float(args.growth_flux_threshold),
                max_additions=int(args.max_additions),
                cooldown_steps=int(args.cooldown_steps),
            )
            arms[result["arm"]] = result
            summary = result["summary"]
            print(
                f"   {result['arm']:<8} min_fid={summary['min_fidelity']:.6f}  "
                f"final_fid={summary['final_fidelity']:.6f}  "
                f"max_flux={summary['max_escape_flux']:.2e}  "
                f"dens_err={summary['max_staggered_density_abs_error']:.2e}  "
                f"growth={summary['growth_event_count']}",
                flush=True,
            )

        regimes_payload[regime] = {
            "u": float(u),
            "g_ep": float(g_ep),
            "n_ph_max": int(n_ph_max),
            "omega": omega,
            "omega_selection_rule": "max_frozen_escape_flux_over_sweep",
            "drive": dict(drive),
            "initial_support_size": len(support),
            "initial_support_total_2q": float(sum(costs[int(i)] for i in support)),
            "growth_flux_threshold": float(args.growth_flux_threshold),
            "arms": arms,
        }
        del h_full, d_full, pool_images

    payload = {
        "schema_version": "paper_iii_qse_adaptive_dynamics_v1",
        "policy": "diagnostic_only_adaptive_dynamics",
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "post_run_diagnostic_only": True,
        },
        "target_roots": _TARGET_ROOTS,
        "cost_weights_preset": "two_qubit_only_v1",
        "growth_rule": (
            "escape_flux > threshold triggers admission of the pool candidate whose "
            "manifold-orthogonal component best aligns with the unrepresented drift; "
            "support rebuilt, state re-injected"
        ),
        "source_sweep_json": str(args.sweep_json),
        "regimes": regimes_payload,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\noutput_json: {args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
