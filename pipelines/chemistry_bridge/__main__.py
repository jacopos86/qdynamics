"""Command line for the Doron chemistry-quantum bridge.

    python -m pipelines.chemistry_bridge probe-backends
    python -m pipelines.chemistry_bridge export-qdynamics --h5 <prefix>_ele.h5 --out problem.json [--frozen-core 1]
    python -m pipelines.chemistry_bridge export-synthetic --kind contract4 --out problem.json [--frozen-core 0]
    python -m pipelines.chemistry_bridge run --manifest problem.manifest.json --out-dir out/ [--backend torch:mps:float32] [--methods mapping,exact,qse,adapt,apm]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from pipelines.chemistry_bridge.array_backend import probe_backends
from pipelines.chemistry_bridge.methods import DEFAULT_METHODS, run_bridge
from pipelines.chemistry_bridge.qdynamics_export import (
    BridgeDependencyUnavailable,
    FrozenCoreReduction,
    QdynamicsConventionError,
    bridge_problem_from_qdynamics,
    load_qdynamics_matrix_elements_h5,
    qdynamics_matrix_elements_from_mapping,
    write_bridge_problem_json,
)
from pipelines.chemistry_bridge.synthetic_witnesses import SYNTHETIC_KINDS, synthetic_qdynamics_mapping


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipelines.chemistry_bridge", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    probe = sub.add_parser("probe-backends", help="report which array backends/devices/dtypes can be exercised here")
    probe.add_argument("--json", action="store_true")
    export = sub.add_parser("export-qdynamics", help="convert a qdynamics <prefix>_ele.h5 export into a strict-intake problem document")
    export.add_argument("--h5", required=True)
    export.add_argument("--out", required=True)
    export.add_argument("--frozen-core", type=int, default=0, help="freeze the lowest N occupied orbitals (declared nuclear_plus_frozen_core)")
    synthetic = sub.add_parser("export-synthetic", help="write a labelled synthetic witness in the producer's schema (not a molecule)")
    synthetic.add_argument("--kind", choices=SYNTHETIC_KINDS, default="contract4")
    synthetic.add_argument("--out", required=True)
    synthetic.add_argument("--frozen-core", type=int, default=0)
    synthetic.add_argument("--seed", type=int, default=20261019)
    run = sub.add_parser("run", help="strict intake + method runners with receipts")
    run.add_argument("--manifest", required=True)
    run.add_argument("--out-dir", required=True)
    run.add_argument("--backend", default="numpy:float64")
    run.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    run.add_argument("--apm-times", default="0,0.02,0.04,0.06,0.08,0.10")
    run.add_argument("--apm-pool", default="full_meta")
    run.add_argument("--apm-parameterization", default="logical_shared", choices=("logical_shared", "per_pauli_term"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "probe-backends":
        reports = probe_backends()
        if args.json:
            print(json.dumps([r.to_json_dict() for r in reports], indent=2))
        else:
            for r in reports:
                print(f"{r.spec:24s} {'available' if r.available else 'UNAVAILABLE'}  exercised={r.exercised_device}  gpu={r.is_gpu}  {r.reason or ''}")
        return 0
    if args.command in {"export-qdynamics", "export-synthetic"}:
        try:
            if args.command == "export-qdynamics":
                elements = load_qdynamics_matrix_elements_h5(args.h5)
            else:
                elements = qdynamics_matrix_elements_from_mapping(synthetic_qdynamics_mapping(args.kind, seed=int(args.seed)), source={"payload_kind": "synthetic_witness", "kind": str(args.kind), "seed": int(args.seed)})
            reduction = FrozenCoreReduction(int(args.frozen_core)) if int(args.frozen_core) > 0 else None
            bridge_problem = bridge_problem_from_qdynamics(elements, frozen_core=reduction)
        except (BridgeDependencyUnavailable, QdynamicsConventionError) as exc:
            print(f"export refused: {exc}", file=sys.stderr)
            return 2
        manifest = write_bridge_problem_json(bridge_problem, args.out)
        print(json.dumps({k: manifest[k] for k in ("problem_json_path", "manifest_path", "raw_sha256", "n_qubits", "num_particles", "reduction")}, indent=2))
        return 0
    if args.command == "run":
        times = [float(v) for v in str(args.apm_times).split(",") if v.strip()]
        summary = run_bridge(
            args.manifest,
            backend_spec=str(args.backend),
            methods=[m for m in str(args.methods).split(",") if m.strip()],
            out_dir=Path(args.out_dir),
            apm_times=times,
            apm_pool_key=str(args.apm_pool),
            apm_parameterization_mode=str(args.apm_parameterization),
        )
        print(json.dumps({"status_by_method": summary["status_by_method"], "out_path": summary.get("out_path"), "backend": summary["backend_spec"]}, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
