#!/usr/bin/env python3
"""Build one Hubbard--Holstein regime seed with the frozen three-layer VQE.

This runner reuses the prospective HH seed worker's frozen construction
(``three_complete_ordered_hh_hamiltonian_layers_v2``: three identical complete
ordered Hamiltonian parent layers, four seeded L-BFGS-B restarts with the exact
compiled adjoint gradient, and the hard ``|E - E_0| <= 1e-6`` ground-state
gate) for an arbitrary declared regime ``(U, g_ep, n_ph_max)``.  It does not run
the conditioning-stress architecture search, fits no driven snapshots, and never
passes exact data to the optimizer.

The artifact it writes is the worker's runtime-loadable payload (an
``adapt_vqe`` block in the ``per_pauli_term`` convention), so the dynamics
runner accepts it through ``--artifact-json`` exactly as it accepts the
prospective lane's seeds.  A seed that fails the gate is never written under
the requested path: the receipt records the failure and the process exits 3,
so ``paper_ii_runs.resolve_regime`` keeps raising ``SeedNotBuiltError``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.time_dynamics import prospective_hh_seed_worker as worker
from pipelines.time_dynamics.prospective_campaign_bootstrap import SeedIdentity
from pipelines.time_dynamics.prospective_execution_bindings import (
    ProspectiveExecutionError,
    SEED_ENERGY_GATE,
)
from pipelines.time_dynamics.run_lock import resolve_code_revision

RUNNER_SCHEMA_V1 = "paper_ii_regime_seed_runner_v1"
SEED_ARTIFACT_SCHEMA_V1 = "paper_ii_regime_seed_artifact_v1"
SEED_RECEIPT_SCHEMA_V1 = "paper_ii_regime_seed_receipt_v1"
DEFAULT_LANE_ID = "paper_ii_regime_seeds_v1"
SEED_POLICY = "generator_derived_fixed_structure_vqe_per_regime_v1"
SEED_INITIALIZER = "generator_derived_fixed_structure_vqe_v1"
SEED_RESTARTS = 4
EXIT_UNQUALIFIED = 3


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seal(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    core = {k: v for k, v in payload.items() if k != key}
    return {**core, key: _canonical_sha(core)}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.part")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def build_seed_identity(
    *,
    regime_id: str,
    u: float,
    g_ep: float,
    n_ph_max: int,
    optimizer_seed: int,
    artifact_path: str,
    lane_id: str = DEFAULT_LANE_ID,
    num_sites: int = 2,
    t: float = 1.0,
    omega0: float = 1.0,
    dv: float = 0.0,
    n_up: int = 1,
    n_dn: int = 1,
    boson_encoding: str = "binary",
    ordering: str = "blocked",
    boundary: str = "open",
) -> SeedIdentity:
    """Declare one regime seed in the prospective worker's physical vocabulary."""

    if int(n_ph_max) not in (1, 3, 7):
        raise ValueError("n_ph_max must fill the binary phonon register (1, 3, or 7)")
    if not str(regime_id).strip():
        raise ValueError("regime_id must be nonempty")
    physical = {
        "lattice_sites": int(num_sites),
        "t_hop": float(t),
        "U": float(u),
        "staggered_potential": float(dv),
        "omega0": float(omega0),
        "g": float(g_ep),
        "n_ph_max": int(n_ph_max),
        "boson_encoding": str(boson_encoding),
        "ordering": str(ordering),
        "boundary": str(boundary),
        "n_up": int(n_up),
        "n_down": int(n_dn),
    }
    return SeedIdentity(
        seed_id=f"{lane_id}::seed::{regime_id}",
        lane_id=str(lane_id),
        model="hubbard_holstein",
        physical=physical,
        policy=SEED_POLICY,
        energy_gate=float(SEED_ENERGY_GATE),
        artifact_path=str(artifact_path),
        external_artifact_sha256=None,
        initializer=SEED_INITIALIZER,
        restarts=SEED_RESTARTS,
        optimizer_seed=int(optimizer_seed),
    )


def build_work_order(seed: SeedIdentity) -> worker.HHSeedWorkOrder:
    """Bind the frozen architecture policy; ``compute_seed`` reads nothing else."""

    policy = worker.architecture_policy()
    return worker.HHSeedWorkOrder(
        request=None,  # type: ignore[arg-type]
        seed=seed,
        source_epoch_receipt_path="",
        source_epoch_receipt_sha256="",
        architecture=policy,
        work_order_sha256=_canonical_sha(
            {"seed_identity": seed.to_json_dict(), "architecture_policy": policy}
        ),
    )


def _artifact_payload(
    *,
    regime_id: str,
    seed: SeedIdentity,
    result: Any,
    code_revision: str,
    order: worker.HHSeedWorkOrder,
) -> dict[str, Any]:
    payload = dict(result.runtime_payload)
    payload.update(
        {
            "schema": SEED_ARTIFACT_SCHEMA_V1,
            "regime_id": str(regime_id),
            "seed_id": seed.seed_id,
            "seed_identity_sha256": seed.identity_sha256,
            "lane_id": seed.lane_id,
            "model": seed.model,
            "physical": dict(seed.physical),
            "seed_policy": seed.policy,
            "initializer": seed.initializer,
            "restarts": seed.restarts,
            "optimizer_seed": seed.optimizer_seed,
            "variational_energy": float(result.variational_energy),
            "exact_energy": float(result.exact_energy),
            "exact_state_infidelity": float(result.exact_state_infidelity),
            "energy_error": abs(float(result.variational_energy) - float(result.exact_energy)),
            "energy_gate": float(SEED_ENERGY_GATE),
            "architecture_binding": dict(result.architecture_binding),
            "optimizer_receipt": dict(result.optimizer_receipt),
            "work_order_sha256": order.work_order_sha256,
            "code_revision": str(code_revision),
            "conditioning_search_used": False,
            "snapshot_conditioning_used": False,
            "qiskit_compilation_used": False,
        }
    )
    return _seal(payload, "seed_artifact_sha256")


def build_regime_seed(
    *,
    regime_id: str,
    seed: SeedIdentity,
    output_json: str | Path,
    repository_root: str | Path = ".",
    compute=worker.compute_seed,
) -> int:
    """Run the frozen VQE, write artifact + receipt, return the process exit code."""

    output = Path(output_json)
    receipt_path = output.with_name("seed_receipt.json")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite an existing seed artifact: {output}")
    order = build_work_order(seed)
    code_revision = resolve_code_revision(repository_root)
    started = time.time()
    receipt: dict[str, Any] = {
        "schema": SEED_RECEIPT_SCHEMA_V1,
        "runner": RUNNER_SCHEMA_V1,
        "regime_id": str(regime_id),
        "seed_id": seed.seed_id,
        "seed_identity_sha256": seed.identity_sha256,
        "physical": dict(seed.physical),
        "optimizer_seed": seed.optimizer_seed,
        "restarts": seed.restarts,
        "energy_gate": float(SEED_ENERGY_GATE),
        "architecture_policy_sha256": order.architecture["architecture_policy_sha256"],
        "work_order_sha256": order.work_order_sha256,
        "code_revision": str(code_revision),
        "conditioning_search_used": False,
        "snapshot_conditioning_used": False,
    }
    try:
        result = compute(order)
    except ProspectiveExecutionError as exc:
        receipt.update(
            {
                "status": "unqualified",
                "failure": str(exc),
                "wall_seconds": round(time.time() - started, 3),
                "artifact": None,
            }
        )
        _write_json(receipt_path, _seal(receipt, "seed_receipt_sha256"))
        print(f"status=unqualified regime_id={regime_id} failure={exc}")
        print(f"receipt_json={receipt_path}")
        return EXIT_UNQUALIFIED
    error = abs(float(result.variational_energy) - float(result.exact_energy))
    if not error <= float(SEED_ENERGY_GATE):
        raise ProspectiveExecutionError(
            f"compute returned an unqualified seed without raising: {error:.16g}"
        )
    artifact = _artifact_payload(
        regime_id=regime_id,
        seed=seed,
        result=result,
        code_revision=code_revision,
        order=order,
    )
    _write_json(output, artifact)
    receipt.update(
        {
            "status": "qualified",
            "artifact": {
                "path": str(output),
                "sha256": _file_sha(output),
                "bytes": output.stat().st_size,
                "seed_artifact_sha256": artifact["seed_artifact_sha256"],
            },
            "energy_error": error,
            "variational_energy": float(result.variational_energy),
            "exact_energy": float(result.exact_energy),
            "exact_state_infidelity": float(result.exact_state_infidelity),
            "num_parameters": int(artifact["adapt_vqe"]["num_parameters"]),
            "wall_seconds": round(time.time() - started, 3),
        }
    )
    _write_json(receipt_path, _seal(receipt, "seed_receipt_sha256"))
    print(f"status=qualified regime_id={regime_id}")
    print(f"energy_error={error:.6e} exact_energy={float(result.exact_energy):.12f}")
    print(f"num_parameters={receipt['num_parameters']}")
    print(f"artifact_json={output}")
    print(f"artifact_sha256={receipt['artifact']['sha256']}")
    print(f"receipt_json={receipt_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--regime-id", required=True)
    parser.add_argument("--u", type=float, required=True)
    parser.add_argument("--g-ep", type=float, required=True)
    parser.add_argument("--n-ph-max", type=int, required=True)
    parser.add_argument("--optimizer-seed", type=int, required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--lane-id", default=DEFAULT_LANE_ID)
    parser.add_argument("--num-sites", type=int, default=2)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--omega0", type=float, default=1.0)
    parser.add_argument("--dv", type=float, default=0.0)
    parser.add_argument("--n-up", type=int, default=1)
    parser.add_argument("--n-dn", type=int, default=1)
    parser.add_argument("--boson-encoding", default="binary")
    parser.add_argument("--ordering", default="blocked")
    parser.add_argument("--boundary", default="open")
    parser.add_argument("--repo-root", default=".")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seed = build_seed_identity(
        regime_id=args.regime_id,
        u=args.u,
        g_ep=args.g_ep,
        n_ph_max=args.n_ph_max,
        optimizer_seed=args.optimizer_seed,
        artifact_path=args.output_json,
        lane_id=args.lane_id,
        num_sites=args.num_sites,
        t=args.t,
        omega0=args.omega0,
        dv=args.dv,
        n_up=args.n_up,
        n_dn=args.n_dn,
        boson_encoding=args.boson_encoding,
        ordering=args.ordering,
        boundary=args.boundary,
    )
    print(f"seed_id={seed.seed_id} seed_identity_sha256={seed.identity_sha256}")
    return build_regime_seed(
        regime_id=args.regime_id,
        seed=seed,
        output_json=args.output_json,
        repository_root=args.repo_root,
    )


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "EXIT_UNQUALIFIED",
    "SEED_ARTIFACT_SCHEMA_V1",
    "SEED_RECEIPT_SCHEMA_V1",
    "build_parser",
    "build_regime_seed",
    "build_seed_identity",
    "build_work_order",
    "main",
]
