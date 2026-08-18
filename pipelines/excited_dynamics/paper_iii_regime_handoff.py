#!/usr/bin/env python3
"""Three-tier driven-dynamics handoff evidence at the stored-seed point.

Completes the frozen -> adaptive-records -> live-circuit escalation story on
the validated weak-coupling seed artifact with the CURRENT production QSE
support and the worst-escape drive frequency from the driven sweep:

1. residual-stop geometry selection + certified exchange on the stored
   158-element basis (production static configuration) -> QSE result and
   ``qse_spectra_v1`` manifest with matrices;
2. frozen-QSE driven propagation of the selected first-excitation root at
   the supplied drive frequency (identical conventions to the driven
   sweep driver);
3. compact root refit of the same root into an HF -> compact-circuit
   state, scaffold runtime promotion, and adaptive append AP-McLachlan
   propagation under the identical drive (with the matched fixed-support
   McLachlan baseline), via the validated promoted-demo machinery.

All three tiers start from the same QSE root and use the same drive and
grid; the exact sector trajectory is the shared diagnostic reference.
Statevector diagnostics; never feeds controller decisions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.excited_dynamics.paper_iii_advisor_demo import (
    _driven_trajectory,
    _half_filled_sector_indices,
    _match_roots_by_overlap,
    _ritz_states,
)
from pipelines.excited_dynamics.paper_iii_promoted_ap_demo import (
    _drive_config,
    _observable_matrices,
    _run_ap_grid,
    _state_fidelity,
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
from pipelines.qse_spectra.io import (
    basis_elements_from_artifact_source,
    load_polynomial_json,
    load_state_json,
    qse_result_to_manifest,
    write_manifest_json,
)
from pipelines.qse_spectra.record_selection import (
    StaticRecordSelectionConfig,
    select_static_qse_records,
)
from pipelines.scaffold.qse_compact_root_refit import (
    CompactQSERootRefitConfig,
    run_compact_qse_root_refit,
)
from pipelines.scaffold.qse_root_refit import reconstruct_qse_root_target
from pipelines.scaffold.qse_runtime_promotion import (
    QSERuntimePromotionConfig,
    promote_qse_root_refit,
)
from pipelines.scaffold.runtime_loader import load_scaffold_runtime_input_from_payload
from pipelines.time_dynamics.ap_mclachlan.inverse import McLachlanInversePolicy

DEFAULT_SEED = (
    REPO_ROOT
    / "output/diagnostics/paper_iii_hh_advisor_demo_20260802_a005/source_seed.json"
)
DEFAULT_SWEEP = (
    REPO_ROOT
    / "output/diagnostics/paper_iii_driven_dynamics_20260819_v1/driven_dynamics.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output/diagnostics/paper_iii_regime_handoff_20260819_v1"
_Q0_POLICY = QSEBasisVectorPolicy(
    reference_projection="q0", basis_vector_normalization="raw_projected"
)
_TARGET_ROOTS = 6


def _fidelity_summary(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    return {"min": float(np.min(values)), "final": float(values[-1])}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-seed-json", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--sweep-json", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--sweep-regime", default="weak_weak")
    parser.add_argument("--omega", type=float, default=None, help="Overrides the sweep worst-escape frequency.")
    parser.add_argument("--drive-amplitude", type=float, default=0.2)
    parser.add_argument("--drive-tbar", type=float, default=4.0)
    parser.add_argument("--drive-phi", type=float, default=0.0)
    parser.add_argument("--t-final", type=float, default=8.0)
    parser.add_argument("--num-steps", type=int, default=160)
    parser.add_argument("--dts", default="0.05", help="Comma-separated AP time steps.")
    parser.add_argument("--residual-stop", type=float, default=1.0e-3)
    parser.add_argument("--max-rounds", type=int, default=30)
    parser.add_argument("--max-selected-paulis", type=int, default=40)
    parser.add_argument("--target-infidelity", type=float, default=1.0e-8)
    parser.add_argument("--refit-maxiter", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "qse_epsstop_exchange_manifest.json"
    refit_path = output_dir / "qse_compact_root_refit.json"
    promoted_path = output_dir / "qse_runtime_promoted_ansatz.json"
    result_path = output_dir / "regime_handoff_result.json"

    if args.omega is not None:
        omega = float(args.omega)
        omega_source = "cli_override"
    else:
        sweep = json.loads(Path(args.sweep_json).read_text(encoding="utf-8"))
        worst = max(
            sweep["regimes"][str(args.sweep_regime)]["omega_sweep"],
            key=lambda row: float(row["summary"]["max_escape_flux"]),
        )
        omega = float(worst["omega"])
        omega_source = f"sweep_worst_escape_{args.sweep_regime}"
    print(f"drive omega = {omega:.4f} ({omega_source})", flush=True)

    seed_path = Path(args.source_seed_json)
    hamiltonian_poly, _ham_prov = load_polynomial_json(seed_path)
    terms = list(hamiltonian_poly.return_polynomial())
    nq = int(terms[0].nqubit())
    prepared_state, _state_prov = load_state_json(seed_path, expected_nq=nq, state_key="initial_state")
    basis, basis_provenance = basis_elements_from_artifact_source(
        seed_path,
        nq=nq,
        hamiltonian=hamiltonian_poly,
        source="full_meta",
        include_hamiltonian_terms=True,
        canonical_hh_full_meta=True,
    )
    print(f"seed loaded: nq={nq}, basis={len(basis)}", flush=True)

    cost_rows = annotate_basis_with_compiled_costs(
        basis,
        num_qubits=nq,
        oracle_kind=ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
        cost_weights=resolve_cost_weights_preset("two_qubit_only_v1"),
    )
    costs = tuple(row.scalarized_canonical_cost for row in cost_rows)
    selection = select_static_qse_records(
        basis,
        config=StaticRecordSelectionConfig(
            mode="geometry_selected",
            max_records=len(basis),
            geometry_target_roots=_TARGET_ROOTS,
            geometry_cost_discount_alpha=1.0,
            geometry_residual_stop=float(args.residual_stop),
        ),
        hamiltonian=hamiltonian_poly,
        prepared_state=prepared_state,
        basis_vector_policy=_Q0_POLICY,
        compiled_costs=costs,
    )
    exchange = run_qse_exchange_maintenance(
        basis,
        selection.selected_original_indices,
        costs,
        hamiltonian=hamiltonian_poly,
        prepared_state=prepared_state,
        basis_vector_policy=_Q0_POLICY,
        config=QSEExchangeConfig(
            max_rounds=int(args.max_rounds),
            target_root_count=_TARGET_ROOTS,
            insertion_shortlist_size=16,
        ),
    )
    final_indices = list(exchange.final_indices)
    print(
        f"support k={len(final_indices)} "
        f"(stop: {None if selection.geometry_stop is None else selection.geometry_stop.get('stop_reason')}, "
        f"exchange patches: {sum(1 for r in exchange.rounds if r['committed_patch'] is not None)})",
        flush=True,
    )

    qse_result = compute_qse_spectra(
        hamiltonian_poly,
        prepared_state,
        tuple(basis[int(i)] for i in final_indices),
        basis_vector_policy=_Q0_POLICY,
    )
    manifest = qse_result_to_manifest(
        qse_result,
        input_provenance={
            "hamiltonian": _ham_prov,
            "state": _state_prov,
            "operator_basis": basis_provenance,
            "selection": {
                "mode": "geometry_selected_residual_stop_plus_exchange_dominance",
                "residual_stop": float(args.residual_stop),
                "selected_original_indices": [int(i) for i in final_indices],
            },
        },
        settings_provenance={
            "run_class": "diagnostic",
            "basis_source": "canonical_hh_full_meta_plus_hamiltonian_terms",
        },
        include_matrices=True,
    )
    write_manifest_json(manifest_path, manifest)

    hamiltonian_full, density_full, phonon_full, sector_indices = _observable_matrices(
        source_seed_json=seed_path
    )
    h_sector = hamiltonian_full[np.ix_(sector_indices, sector_indices)]
    exact_energies, exact_vectors = np.linalg.eigh(0.5 * (h_sector + h_sector.conj().T))
    ritz = _ritz_states(qse_result)
    matches = _match_roots_by_overlap(
        ritz, exact_vectors, np.asarray(sector_indices, dtype=int), root_count=1
    )
    matched_exact_index = int(matches[0])
    root0 = ritz[0]

    print("frozen arm ...", flush=True)
    frozen_rows, frozen_metrics = _driven_trajectory(
        result=qse_result,
        initial_root_state=root0,
        exact_matched_vector=exact_vectors[:, matched_exact_index],
        sector_indices=np.asarray(sector_indices, dtype=int),
        hamiltonian_full=hamiltonian_full,
        drive_full=density_full,
        phonon_full=phonon_full,
        drive_amplitude=float(args.drive_amplitude),
        drive_omega=omega,
        drive_tbar=float(args.drive_tbar),
        drive_phi=float(args.drive_phi),
        t_final=float(args.t_final),
        num_steps=int(args.num_steps),
    )
    frozen_fid = _fidelity_summary(frozen_rows, "qse_exact_state_fidelity")
    print(f"   frozen min_fid={frozen_fid['min']:.6f} final_fid={frozen_fid['final']:.6f}", flush=True)

    print("compact root refit ...", flush=True)
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    target, _prepared, _basis, _target_nq = reconstruct_qse_root_target(
        manifest_payload,
        qse_result_json=manifest_path,
        state_index=0,
        allow_ground_state=False,
        amplitude_cutoff=1.0e-12,
    )
    target_check = _state_fidelity(target.state, root0)
    if target_check < 1.0 - 1.0e-10:
        raise ValueError(f"Reconstructed manifest root differs from in-process root: fid={target_check}")
    root_refit = run_compact_qse_root_refit(
        CompactQSERootRefitConfig(
            qse_result_json=manifest_path,
            state_index=0,
            output_json=refit_path,
            base_scaffold_json=seed_path,
            hamiltonian_json=seed_path,
            max_selected_paulis=int(args.max_selected_paulis),
            target_infidelity=float(args.target_infidelity),
            max_energy_error=1.0e-6,
            max_physical_residual=1.0e-3,
            optimizer_maxiter=int(args.refit_maxiter),
        )
    )
    print(
        f"   refit achieved infidelity {root_refit['fit_summary']['infidelity']:.3e} "
        f"with {root_refit['compact_refit']['selected_pauli_count']} paulis",
        flush=True,
    )
    promotion = promote_qse_root_refit(
        QSERuntimePromotionConfig(
            qse_root_refit_json=refit_path,
            output_json=promoted_path,
            runtime_template_json=seed_path,
            require_runtime_contract=True,
            max_reconstruction_error=1.0e-10,
        )
    )
    runtime_input = load_scaffold_runtime_input_from_payload(
        promotion["runtime_payload"], artifact_json=promoted_path
    )
    promoted_fidelity = _state_fidelity(runtime_input.psi_initial, target.state)
    print(f"   promoted runtime fidelity vs root: {promoted_fidelity:.12f}", flush=True)

    controller_drive = {
        "amplitude": float(args.drive_amplitude),
        "omega": omega,
        "tbar": float(args.drive_tbar),
        "phi": float(args.drive_phi),
        "t_final": float(args.t_final),
    }
    inverse_policy = McLachlanInversePolicy(
        pinv_rcond=1.0e-10, ridge_lambda=1.0e-7, solve_damping=0.0
    )
    grids: dict[str, Any] = {}
    def _progress(payload: Any) -> None:
        index = int(payload.get("index", -1))
        if payload.get("phase") == "checkpoint_start" or index % 10 == 0:
            print(
                f"   [ap-progress] i={index} t={payload.get('time'):.3f} "
                f"params={payload.get('runtime_parameter_count')} "
                f"rr={payload.get('mclachlan_residual_ratio'):.2e}",
                flush=True,
            )

    for dt in sorted({float(v) for v in str(args.dts).split(",")}, reverse=True):
        print(f"AP-McLachlan arm (dt={dt:g}) ...", flush=True)
        grid, _states = _run_ap_grid(
            runtime_input=runtime_input,
            source_seed_json=seed_path,
            drive_config=_drive_config(controller_drive),
            drive=controller_drive,
            dt=float(dt),
            initial_target_full=target.state,
            hamiltonian_full=hamiltonian_full,
            density_full=density_full,
            phonon_full=phonon_full,
            sector_indices=np.asarray(sector_indices, dtype=int),
            inverse_policy=inverse_policy,
            progress_callback=_progress,
        )
        ap_fid = _fidelity_summary(grid["trajectory"], "ap_exact_state_fidelity")
        baseline_fid = _fidelity_summary(
            grid["fixed_support_baseline"]["trajectory"], "ap_exact_state_fidelity"
        )
        print(
            f"   adaptive AP min_fid={ap_fid['min']:.6f} final_fid={ap_fid['final']:.6f} "
            f"(params {grid['metrics']['initial_runtime_parameter_count']}->"
            f"{grid['metrics']['final_runtime_parameter_count']}, "
            f"{grid['metrics']['accepted_support_patch_count']} patches); "
            f"fixed baseline min_fid={baseline_fid['min']:.6f}",
            flush=True,
        )
        grids[f"dt_{dt:g}"] = grid

    payload = {
        "schema_version": "paper_iii_regime_handoff_v1",
        "policy": "diagnostic_only_three_tier_handoff",
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "post_run_diagnostic_only": True,
        },
        "drive": {**controller_drive, "omega_source": omega_source,
                  "operator": "staggered_density",
                  "waveform": "gaussian_sinusoid A*sin(w t + phi)*exp(-t^2/(2 tbar^2))"},
        "source_seed_json": str(seed_path),
        "qse_manifest_json": str(manifest_path),
        "selection": {
            "support_size": len(final_indices),
            "selected_original_indices": [int(i) for i in final_indices],
            "total_2q": float(sum(costs[int(i)] for i in final_indices)),
            "geometry_stop": None if selection.geometry_stop is None else dict(selection.geometry_stop),
        },
        "root_refit": {
            "achieved_infidelity": float(root_refit["fit_summary"]["infidelity"]),
            "selected_pauli_count": int(root_refit["compact_refit"]["selected_pauli_count"]),
            "promoted_runtime_fidelity_vs_root": float(promoted_fidelity),
            "refit_json": str(refit_path),
            "promoted_json": str(promoted_path),
        },
        "frozen_arm": {
            "trajectory": frozen_rows,
            "metrics": frozen_metrics,
            "fidelity": frozen_fid,
        },
        "ap_grids": grids,
        "summary": {
            "frozen_min_fidelity": frozen_fid["min"],
            "ap_adaptive_min_fidelity": {
                key: _fidelity_summary(grid["trajectory"], "ap_exact_state_fidelity")["min"]
                for key, grid in grids.items()
            },
            "ap_fixed_baseline_min_fidelity": {
                key: _fidelity_summary(
                    grid["fixed_support_baseline"]["trajectory"], "ap_exact_state_fidelity"
                )["min"]
                for key, grid in grids.items()
            },
        },
    }

    def _json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(v) for v in value]
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, np.ndarray):
            return [_json_safe(v) for v in value.tolist()]
        return value

    result_path.write_text(
        json.dumps(_json_safe(payload), indent=1, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"\noutput_json: {result_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
