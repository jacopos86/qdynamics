from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.quantum.ansatz_parameterization import deserialize_layout, expand_legacy_logical_theta
from src.quantum.chemistry.molecular_adapt_core import run_pipeline_local_adapt_vqe_with_pool
from src.quantum.chemistry.vibronic_h2 import (
    _boson_code_bits,
    _dense_matrix_from_polynomial,
    _fermion_sector_bits,
    build_vibronic_h2_model,
)


@dataclass(frozen=True)
class TrialSummary:
    removed_family: str | None
    keep_families: tuple[str, ...]
    operators: tuple[str, ...]
    abs_delta_e: float
    exact_fidelity: float
    ansatz_depth: int
    logical_parameter_count: int
    runtime_parameter_count: int
    logical_circuit_size: int
    logical_circuit_depth: int
    logical_two_qubit_count: int
    heavyhex_two_qubit_count: int | None
    heavyhex_depth: int | None
    payload: dict[str, Any]

    def to_jsonable(self) -> dict[str, Any]:
        out = {
            "removed_family": self.removed_family,
            "keep_families": list(self.keep_families),
            "operators": list(self.operators),
            "abs_delta_e": float(self.abs_delta_e),
            "exact_fidelity": float(self.exact_fidelity),
            "ansatz_depth": int(self.ansatz_depth),
            "logical_parameter_count": int(self.logical_parameter_count),
            "runtime_parameter_count": int(self.runtime_parameter_count),
            "logical_circuit_size": int(self.logical_circuit_size),
            "logical_circuit_depth": int(self.logical_circuit_depth),
            "logical_two_qubit_count": int(self.logical_two_qubit_count),
            "heavyhex_two_qubit_count": (
                None if self.heavyhex_two_qubit_count is None else int(self.heavyhex_two_qubit_count)
            ),
            "heavyhex_depth": None if self.heavyhex_depth is None else int(self.heavyhex_depth),
            "payload": dict(self.payload),
        }
        return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chemistry-local class-prune redo for vibronic H2.")
    p.add_argument("--bond-length", type=float, default=0.7414)
    p.add_argument("--bond-step", type=float, default=0.01)
    p.add_argument("--basis", type=str, default="sto-3g")
    p.add_argument("--n-ph-max", type=int, default=3)
    p.add_argument("--boson-encoding", choices=["binary", "unary"], default="binary")
    p.add_argument("--coupling-scale", type=float, default=1.0)
    p.add_argument("--delta-threshold", type=float, default=5e-4)
    p.add_argument("--fidelity-drop-tol", type=float, default=1e-4)
    p.add_argument("--adapt-max-depth", type=int, default=8)
    p.add_argument("--adapt-eps-grad", type=float, default=1e-8)
    p.add_argument("--adapt-eps-energy", type=float, default=1e-8)
    p.add_argument("--adapt-maxiter", type=int, default=400)
    p.add_argument("--adapt-inner-optimizer", choices=["COBYLA", "POWELL", "SPSA"], default="COBYLA")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output-json", type=str, default=None)
    return p.parse_args(argv)


def _classify_pool_family(label: str) -> str:
    label_str = str(label)
    if label_str.startswith("el::uccsd_sing("):
        return "el_uccsd_sing"
    if label_str.startswith("el::uccsd_dbl("):
        return "el_uccsd_dbl"
    if label_str == "boson::p":
        return "boson_p"
    if label_str == "coupled::dH_dR_times_p":
        return "coupled_dH_dR_p"
    if label_str.startswith("coupled::occ::") and label_str.endswith("::p"):
        return "coupled_occ_p"
    if label_str.startswith("coupled::el::uccsd_sing(") and label_str.endswith("::p"):
        return "coupled_el_uccsd_sing_p"
    if label_str.startswith("coupled::el::uccsd_dbl(") and label_str.endswith("::p"):
        return "coupled_el_uccsd_dbl_p"
    return f"other::{label_str}"


def _exact_ground_state_physical_sector(
    h_poly: Any,
    *,
    n_spatial_orbitals: int,
    num_particles: tuple[int, int],
    n_ph_max: int,
    boson_encoding: str,
) -> tuple[float, np.ndarray]:
    dense = _dense_matrix_from_polynomial(h_poly)
    n_fermion_qubits = 2 * int(n_spatial_orbitals)
    fermion_bits = _fermion_sector_bits(
        n_spatial_orbitals=int(n_spatial_orbitals),
        num_particles=tuple(int(x) for x in num_particles),
    )
    boson_bits = _boson_code_bits(n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding))
    basis_indices = [int(f_bits + (b_bits << n_fermion_qubits)) for b_bits in boson_bits for f_bits in fermion_bits]
    sub = dense[np.ix_(basis_indices, basis_indices)]
    evals, evecs = np.linalg.eigh(sub)
    ground_idx = int(np.argmin(np.real(evals)))
    ground_energy = float(np.real(evals[ground_idx]))
    ground_sub = np.asarray(evecs[:, ground_idx], dtype=complex)
    ground_full = np.zeros(dense.shape[0], dtype=complex)
    ground_full[np.asarray(basis_indices, dtype=int)] = ground_sub
    ground_full = np.asarray(ground_full / np.linalg.norm(ground_full), dtype=complex)
    return ground_energy, ground_full


def _logical_circuit_stats(payload: dict[str, Any], psi_ref: np.ndarray) -> dict[str, int | None]:
    layout = deserialize_layout(payload["parameterization"])
    theta_logical = np.asarray(payload["logical_optimal_point"], dtype=float)
    theta_runtime = expand_legacy_logical_theta(theta_logical, layout)
    try:
        from pipelines.hardcoded.adapt_circuit_execution import build_ansatz_circuit
    except Exception:
        return {
            "logical_circuit_size": None,
            "logical_circuit_depth": None,
            "logical_two_qubit_count": None,
            "heavyhex_two_qubit_count": None,
            "heavyhex_depth": None,
        }

    qc = build_ansatz_circuit(layout, theta_runtime, int(layout.blocks[0].terms[0].nq), ref_state=psi_ref)
    logical_two_qubit = int(sum(1 for inst in qc.data if len(inst.qubits) == 2))
    out: dict[str, int | None] = {
        "logical_circuit_size": int(qc.size()),
        "logical_circuit_depth": int(qc.depth() or 0),
        "logical_two_qubit_count": int(logical_two_qubit),
        "heavyhex_two_qubit_count": None,
        "heavyhex_depth": None,
    }
    try:
        from qiskit import transpile
        from qiskit.transpiler import CouplingMap

        compiled = transpile(
            qc,
            basis_gates=["rz", "sx", "x", "ecr"],
            coupling_map=CouplingMap.from_heavy_hex(distance=3),
            optimization_level=3,
            seed_transpiler=7,
            layout_method="sabre",
            routing_method="sabre",
        )
        out["heavyhex_two_qubit_count"] = int(sum(1 for inst in compiled.data if len(inst.qubits) == 2))
        out["heavyhex_depth"] = int(compiled.depth() or 0)
    except Exception:
        pass
    return out


def _run_trial(
    *,
    pool: Sequence[Any],
    keep_families: Sequence[str],
    removed_family: str | None,
    model: Any,
    exact_gs_energy: float,
    exact_ground_state: np.ndarray,
    args: argparse.Namespace,
) -> TrialSummary:
    result = run_pipeline_local_adapt_vqe_with_pool(
        h_poly=model.h_vibronic,
        psi_ref=model.psi_ref,
        pool=list(pool),
        exact_gs_energy=float(exact_gs_energy),
        max_depth=int(args.adapt_max_depth),
        eps_grad=float(args.adapt_eps_grad),
        eps_energy=float(args.adapt_eps_energy),
        maxiter=int(args.adapt_maxiter),
        optimizer=str(args.adapt_inner_optimizer),
        seed=int(args.seed),
        pool_type="vibronic_h2_class_prune_pool",
        metadata={"num_particles": {"n_up": 1, "n_dn": 1}},
        parameterization_mode="logical_shared",
        reopt_policy="windowed",
        window_size=3,
        window_topk=0,
        full_refit_every=0,
    )
    fidelity = float(abs(np.vdot(np.asarray(exact_ground_state, dtype=complex), np.asarray(result.psi_final, dtype=complex))) ** 2)
    circuit_stats = _logical_circuit_stats(result.payload, model.psi_ref)
    return TrialSummary(
        removed_family=(None if removed_family is None else str(removed_family)),
        keep_families=tuple(str(x) for x in keep_families),
        operators=tuple(str(x) for x in result.payload.get("operators", [])),
        abs_delta_e=float(result.payload["abs_delta_e"]),
        exact_fidelity=float(fidelity),
        ansatz_depth=int(result.payload["ansatz_depth"]),
        logical_parameter_count=int(result.payload["logical_num_parameters"]),
        runtime_parameter_count=int(result.payload["parameterization"]["runtime_parameter_count"]),
        logical_circuit_size=int(circuit_stats["logical_circuit_size"] or 0),
        logical_circuit_depth=int(circuit_stats["logical_circuit_depth"] or 0),
        logical_two_qubit_count=int(circuit_stats["logical_two_qubit_count"] or 0),
        heavyhex_two_qubit_count=(
            None if circuit_stats["heavyhex_two_qubit_count"] is None else int(circuit_stats["heavyhex_two_qubit_count"])
        ),
        heavyhex_depth=None if circuit_stats["heavyhex_depth"] is None else int(circuit_stats["heavyhex_depth"]),
        payload=dict(result.payload),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    model = build_vibronic_h2_model(
        bond_length_angstrom=float(args.bond_length),
        bond_step_angstrom=float(args.bond_step),
        basis=str(args.basis),
        n_ph_max=int(args.n_ph_max),
        boson_encoding=str(args.boson_encoding),
        coupling_scale=float(args.coupling_scale),
        ordering="blocked",
    )
    exact_gs_energy, exact_ground_state = _exact_ground_state_physical_sector(
        model.h_vibronic,
        n_spatial_orbitals=2,
        num_particles=(1, 1),
        n_ph_max=int(model.n_ph_max),
        boson_encoding=str(model.boson_encoding),
    )

    family_order: list[str] = []
    seen: set[str] = set()
    for term in model.pool:
        family = _classify_pool_family(str(term.label))
        if family in seen:
            continue
        seen.add(family)
        family_order.append(family)

    parent_trial = _run_trial(
        pool=model.pool,
        keep_families=tuple(family_order),
        removed_family=None,
        model=model,
        exact_gs_energy=float(exact_gs_energy),
        exact_ground_state=exact_ground_state,
        args=args,
    )
    fidelity_floor = float(parent_trial.exact_fidelity - float(args.fidelity_drop_tol))
    delta_threshold = float(args.delta_threshold)

    keep_families = list(family_order)
    accepted_steps: list[dict[str, Any]] = []
    attempted_rounds: list[dict[str, Any]] = []
    current_best = parent_trial

    while True:
        round_trials: list[dict[str, Any]] = []
        accepted_candidates: list[TrialSummary] = []
        for family in list(keep_families):
            reduced_keep = tuple(str(x) for x in keep_families if str(x) != str(family))
            if len(reduced_keep) <= 0:
                continue
            reduced_pool = [term for term in model.pool if _classify_pool_family(str(term.label)) in set(reduced_keep)]
            if not reduced_pool:
                continue
            trial = _run_trial(
                pool=reduced_pool,
                keep_families=reduced_keep,
                removed_family=str(family),
                model=model,
                exact_gs_energy=float(exact_gs_energy),
                exact_ground_state=exact_ground_state,
                args=args,
            )
            accepted = bool(trial.abs_delta_e <= delta_threshold and trial.exact_fidelity >= fidelity_floor)
            round_trials.append({
                **trial.to_jsonable(),
                "accepted": bool(accepted),
                "delta_threshold": float(delta_threshold),
                "fidelity_floor": float(fidelity_floor),
            })
            if accepted:
                accepted_candidates.append(trial)

        attempted_rounds.append({
            "keep_families_before_round": list(keep_families),
            "trials": list(round_trials),
        })
        if not accepted_candidates:
            break

        def _rank_key(trial: TrialSummary) -> tuple[float, int, int, float, str]:
            heavyhex_2q = float("inf") if trial.heavyhex_two_qubit_count is None else float(trial.heavyhex_two_qubit_count)
            return (
                heavyhex_2q,
                int(trial.ansatz_depth),
                int(trial.runtime_parameter_count),
                float(trial.abs_delta_e),
                str(trial.removed_family),
            )

        chosen = min(accepted_candidates, key=_rank_key)
        keep_families = [fam for fam in keep_families if str(fam) != str(chosen.removed_family)]
        current_best = chosen
        accepted_steps.append(
            {
                "removed_family": str(chosen.removed_family),
                "keep_families_after_accept": list(keep_families),
                "accepted_trial": chosen.to_jsonable(),
            }
        )

    chemistry_dir = Path(__file__).resolve().parent
    output_json = Path(args.output_json) if args.output_json not in {None, ""} else (chemistry_dir / "vibronic_h2_class_prune_result.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "prototype": "chemistry_local_vibronic_h2_class_prune_v1",
        "math_reference": "MATH/Math.md §19.3.1 weak-coupling L=2 class-prune redo",
        "accept_rule": {
            "abs_delta_e_le": float(delta_threshold),
            "exact_fidelity_ge": float(fidelity_floor),
            "parent_exact_fidelity": float(parent_trial.exact_fidelity),
            "parent_fidelity_drop_tol": float(args.fidelity_drop_tol),
        },
        "family_order": list(family_order),
        "parent": parent_trial.to_jsonable(),
        "accepted_steps": list(accepted_steps),
        "final": current_best.to_jsonable(),
        "attempted_rounds": list(attempted_rounds),
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote JSON: {output_json}")
    print(
        "Class-prune result: "
        f"parent_depth={parent_trial.ansatz_depth} final_depth={current_best.ansatz_depth} "
        f"parent_abs_delta_e={parent_trial.abs_delta_e:.3e} final_abs_delta_e={current_best.abs_delta_e:.3e} "
        f"parent_fidelity={parent_trial.exact_fidelity:.9f} final_fidelity={current_best.exact_fidelity:.9f}"
    )
    print(f"Final keep families: {list(current_best.keep_families)}")
    print(f"Final operators: {list(current_best.operators)}")


if __name__ == "__main__":
    main()
